import numpy as np
import pandas as pd
import itertools
import pickle
import os
from tqdm import tqdm
import argparse

assert 'NBC_ROOT' in os.environ, 'set NBC_ROOT envvar'
NBC_ROOT = os.environ['NBC_ROOT']

#---onload---
participants = {
    'train': ['1_1a', '2_2a', '5_1c', '6_2c', \
        '7_1a', '8_2a', '9_1b', '10_2b', '11_1c', '12_2c', \
        '13_1a', '14_2a', '15_1b', '16_2b'],
    'dev': ['17_1c', '18_2c'],
    'test': ['3_1b', '4_2b']
}
target_tokens = [
    'pick_VERB',
    'take_VERB',
    'get_VERB',
    'put_VERB',
    'drop_VERB',
    'eat_VERB',
    'wash_VERB',
    'play_VERB',
    'hold_VERB',
    'open_VERB',
    'close_VERB',
    'go_VERB',
    'throw_VERB',
    'walk_VERB',
    'give_VERB',
    'shake_VERB',
    'cook_VERB',
    'stop_VERB',
    'push_VERB'
]

def verify_paths():
    for type in participants.keys():
        for participant, task in itertools.product(participants[type], range(1, 7)):
            path = NBC_ROOT + 'release/{}_task{}/spatial.json'.format(participant, task)
            assert os.path.exists(path), '{} does not exists'.format(path)
verify_paths()
#------

def verify_spatial(df):
    steps = df['step'].unique()
    assert (np.arange(steps[0], steps[-1] + 1) == steps).all()
    objects = df['id'].unique()
    for step, group in df.groupby('step'):
        assert len(group) == len(objects)

def subsample(df, s):
    steps = df['step'].unique()
    mask = steps[np.arange(0, len(steps), s)]
    df = df[df['step'].isin(mask)]
    return df

def get_most_moving_obj(seq):
    max_motion = 0
    most_moving = None
    for id, rows in seq.groupby('id'):
        if rows.iloc[0]['name'] in ['Head', 'LeftHand', 'RightHand']:
            continue
        pos = rows[['posX', 'posY', 'posZ']].to_numpy()
        motion = np.sum(np.var(pos, axis=0))
        if motion > max_motion:
            max_motion = motion
            most_moving = rows.iloc[0]['name']
    return most_moving

def get_most_moving_hand(seq):
    max_motion = 0
    most_moving = None
    for id, rows in seq.groupby('id'):
        if rows.iloc[0]['name'] not in ['LeftHand', 'RightHand']:
            continue
        pos = rows[['posX', 'posY', 'posZ']].to_numpy()
        motion = np.sum(np.var(pos, axis=0))
        if motion > max_motion:
            max_motion = motion
            most_moving = rows.iloc[0]['name']
    return most_moving

def get_feature_direct(seq, target, feat):
    return seq[seq['name'] == target][feat].to_numpy()[:, np.newaxis]

def reparameterize(vec):
    start = vec[0]; end = vec[-1]
    mean = np.mean(vec); var = np.var(vec)
    min = np.min(vec); max = np.max(vec)
    argmin = np.argmin(vec) / 450.; argmax = np.argmax(vec) / 450.
    return np.array([start, end, mean, var, min, max, argmin, argmax])

def dist_to_head(seq, target):
    head_pos = seq[seq['name'] == 'Head'][['posX', 'posY', 'posZ']].to_numpy()
    target_pos = seq[seq['name'] == target][['posX', 'posY', 'posZ']].to_numpy()
    dist = np.sum(np.square(target_pos - head_pos), axis=1)
    return reparameterize(dist)

def avg_vel(seq, target):
    rows = seq[seq['name'] == target]
    avg_vel = np.mean(rows[['velX', 'velY', 'velZ']].to_numpy(), axis=1)
    return reparameterize(avg_vel)

def var_vel(seq, target):
    rows = seq[seq['name'] == target]
    var_vel = np.var(rows[['velX', 'velY', 'velZ']].to_numpy(), axis=1)
    return np.mean(var_vel)[np.newaxis]

def traj(seq, target):
    rows = seq[seq['name'] == target]
    pos = rows[['posX', 'posY', 'posZ']].to_numpy()
    start = pos[0]; end = pos[-1]
    peak = np.max(pos, axis=0); trough = np.min(pos, axis=0)
    peak_idx = np.argmax(pos, axis=0); trough_idx = np.argmin(pos, axis=0)
    feat = []
    for i in range(3):
        feat += [start[i], end[i], peak[i], trough[i], peak_idx[i], trough_idx[i]]
    for i in range(3):
        start = pos[0, i]; end = pos[-1, i]
        if peak_idx[i] < trough_idx[i]:
            kp1 = peak[i]
            kp2 = trough[i]
        else:
            kp1 = trough[i]
            kp2 = peak[i]
        feat += [kp1 - start, kp2 - kp1, end - kp2]
    return np.array(feat)

def avg_rel(seq, target):
    rows = seq[seq['name'] == target]
    avg_pos = rows[['relPosX', 'relPosY', 'relPosZ']].to_numpy()
    return np.mean(avg_pos, axis=0)

def speed(seq, target):
    rows = seq[seq['name'] == target]
    speed = rows.apply(lambda row: row['velX'] * row['velX'] + row['velY'] * row['velY'] + row['velZ'] * row['velZ'], axis=1).to_numpy()
    return speed[:, np.newaxis]

def moving(seq, target):
    rows = seq[seq['name'] == target]
    speed = rows.apply(lambda row: row['velX'] * row['velX'] + row['velY'] * row['velY'] + row['velZ'] * row['velZ'], axis=1).to_numpy()
    speed[speed > 0] = 1
    return speed[:, np.newaxis]

class NBC:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--subsample', help='subsampling step size, i.e. 1 for original 90hz, 9 for 10hz, 90 for 1hz', type=int, default=90)
        parser.add_argument('--dynamic_only', help='filter to only objects that can move', type=bool, default=True)
        parser.add_argument('--train_sequencing', choices=['token_aligned', 'chunked', 'session', 'actions'], default='token_aligned')
        parser.add_argument('--dev_sequencing', choices=['token_aligned', 'chunked', 'session', 'actions'], default='chunked')
        parser.add_argument('--test_sequencing', choices=['token_aligned', 'chunked', 'session', 'actions'], default='chunked')
        parser.add_argument('--features', nargs='+', help='feature:target, e.g. posX:Apple, dist_to_head:most_moving_obj')
        parser.add_argument('--label_method', choices=['nonzero_any', 'nonzero_by_dim', 'actions', 'actions_rhand_apple'], default='nonzero_any')
        parser.add_argument('--trim', help='whether to trim long idle segments', action='store_true')

    def __init__(self, args):
        self.args = args
        self.load()
        self.split_sequences()
        if args.features is not None:
            self.featurize()
            self.generate_labels()
            if args.trim:
                self.trim()

    def featurize(self):
        functions = {
            'one_hot_label': self.one_hot_label,
            'dist_to_head': dist_to_head,
            'avg_vel': avg_vel,
            'var_vel': var_vel,
            'traj': traj,
            'avg_rel': avg_rel,
            'speed': speed,
            'moving': moving
        }

        def parse_target(seq, target):
            if target == 'most_moving_obj':
                return get_most_moving_obj(seq)
            elif target == 'most_moving_hand':
                return get_most_moving_hand(seq)
            else:
                assert target in self.df['train']['name'].unique(), target
                return target

        features = {'train': {}, 'dev': {}, 'test': {}}
        for type in participants.keys():
            for key, seq in self.sequences[type].items():
                n = seq['step'].unique().shape[0]
                features[type][key] = []
                for entry in self.args.features:
                    feature, target = entry.split(':')
                    target = parse_target(seq, target)
                    if target is None:
                        features[type][key].append(np.zeros(n,))
                        continue
                    if feature in functions:
                        feat = functions[feature](seq, target)
                    else:
                        assert feature in seq.columns, feature
                        feat = get_feature_direct(seq, target, feature)
                    assert feat.shape[0] == n or feat.ndim == 1, (feat.shape, n)
                    features[type][key].append(feat)
                if features[type][key][0].ndim == 1:
                    for i in range(len(features[type][key])):
                        assert features[type][key][i].ndim == 1
                    features[type][key] = np.concatenate(features[type][key])
                else:
                    for i in range(len(features[type][key])):
                        assert features[type][key][i].ndim == 2
                    features[type][key] = np.concatenate(features[type][key], axis=-1)

        self.features = features

    def generate_labels(self):
        labels = {'train': {}, 'dev': {}, 'test': {}}
        for type in participants.keys():
            if self.args.label_method == 'actions' or self.args.label_method == 'actions_rhand_apple':
                actions = pd.read_json(NBC_ROOT + 'actions.json', orient='index')
                if self.args.label_method == 'actions_rhand_apple':
                    actions = actions[(actions['target'] == 'Apple') & (actions['hand'] == 'RightHand')]
                for key, steps in self.steps[type].items():
                    session = key[0]
                    group = actions[actions['session'] == session]
                    feat = self.features[type][key]
                    labels_ = np.zeros((feat.shape[0],))
                    actions_ = actions[actions['session'] == session]
                    for _, action in actions_.iterrows():
                        steps_ = np.arange(action['start_step'], action['end_step'])
                        for step in steps_:
                            if step in steps:
                                idx = (steps == step).argmax()
                                labels_[idx] = ['reach', 'pick', 'put', 'retract'].index(action['action']) + 1
                    labels[type][key] = labels_
                self.n_classes = 5
            elif self.args.label_method == 'nonzero_any':
                for key in self.features[type].keys():
                    feat = self.features[type][key]
                    labels_ = np.any(np.abs(feat) > 0, axis=1).astype(int)
                    labels[type][key] = labels_
                self.n_classes = 2
            else:
                assert self.args.label_method == 'nonzero_by_dim'
                for key in self.features[type].keys():
                    feat = self.features[type][key]
                    labels_ = np.zeros((feat.shape[0],))
                    nonzero = (np.abs(feat) > 0)
                    labels_[np.any(nonzero, axis=1)] = (np.argmax(nonzero, axis=1) + 1)[np.any(nonzero, axis=1)]
                    labels[type][key] = labels_
                self.n_classes = next(iter(self.features[type].values())).shape[-1] + 1
        self.labels = labels

    def trim(self):
        for type in participants.keys():
            for key in list(self.features[type].keys()):
                feat = self.features[type][key]
                labels = self.labels[type][key]
                steps = self.steps[type][key]
                if np.all(labels == 0):
                    del self.features[type][key]
                    del self.labels[type][key]
                    del self.steps[type][key]
                else:
                    mask = labels.copy()
                    for i in range(1, 5): #5 frames of trim padding
                        mask += np.roll(labels, i) + np.roll(labels, -i)
                    self.labels[type][key] = labels[mask > 0]
                    self.features[type][key] = feat[mask > 0, :]
                    self.steps[type][key] = steps[mask > 0]

    def split_sequences(self):
        #split dataset into sequences using one of three methods
        sequencing = {'train': self.args.train_sequencing, 'dev': self.args.dev_sequencing, 'test': self.args.test_sequencing}
        sequences = {'train': {}, 'dev': {}, 'test': {}}
        steps = {'train': {}, 'dev': {}, 'test': {}}
        for type in participants.keys():
            if sequencing[type] == 'token_aligned':
                words = pd.read_json(NBC_ROOT + 'words.json', orient='index')
                words['token'] = words.apply(lambda row: row['lemma'] + '_' + row['pos'], axis=1)
                tokens = words[words['token'].isin(target_tokens)]
                for participant, task in itertools.product(participants[type], range(1, 7)):
                    session = '{}_task{}'.format(participant, task)
                    group = self.df[type][self.df[type]['session'] == session]
                    tokens_ = tokens[(tokens['participant'] == participant) & (tokens['task'] == task)]
                    for _, row in tokens_.iterrows():
                        if row['start_step'] + 450 > group.iloc[-1]['step']:
                            continue
                        steps_ = np.arange(row['start_step'], row['start_step'] + 450)
                        rows = group[group['step'].isin(steps_)]
                        assert len(rows) > 0, (group['step'], steps_)
                        sequences[type][(session, row['start_step'], row['token'])] = rows
                        steps[type][(session, row['start_step'], row['token'])] = rows['step'].unique()
            elif sequencing[type] == 'chunked':
                for participant, task in itertools.product(participants[type], range(1, 7)):
                    session = '{}_task{}'.format(participant, task)
                    group = self.df[type][self.df[type]['session'] == session]
                    steps_ = group['step'].unique()
                    for step in np.arange(steps_[0], steps_[-1] - 450, 450):
                        rows = group[group['step'].isin(range(step, step + 450))]
                        assert len(rows) > 0, (group['step'], steps_)
                        sequences[type][(session, step)] = rows
                        steps[type][(session, step)] = rows['step'].unique()
            elif sequencing[type] == 'actions':
                actions = pd.read_json(NBC_ROOT + 'actions.json', orient='index')
                for participant, task in itertools.product(participants[type], range(1, 7)):
                    session = '{}_task{}'.format(participant, task)
                    group = self.df[type][self.df[type]['session'] == session]
                    actions_ = actions[actions['session'] == session]
                    for idx, row in actions_.iterrows():
                        action = row['action']; target = row['target']; hand = row['hand']
                        steps_ = np.arange(row['start_step'], row['end_step'])
                        rows = group[group['step'].isin(steps_)]
                        assert len(rows) > 0, (group['step'], steps_, 'subsampling may be too high')
                        sequences[type][(session, action, target, hand, idx)] = rows
                        steps[type][(session, action, target, hand, idx)] = rows['step'].unique()
            else:
                assert sequencing[type] == 'session'
                for session, group in self.df[type].groupby('session'):
                    sequences[type][(session,)] = group
                    steps[type][(session,)] = group['step'].unique()
        self.sequences = sequences
        self.steps = steps

    def load(self):
        #load from tmp file if possible
        tmp_path = NBC_ROOT + 'tmp/spatial_subsample={}_dynamic-only={}.p'.format(self.args.subsample, self.args.dynamic_only)
        if os.path.exists(tmp_path):
            with open(tmp_path, 'rb') as f:
                self.df = pickle.load(f)
                return

        #otherwise, build dataset from disk and save
        self.df = {'train': [], 'dev': [], 'test': []}
        for type in participants.keys():
            print('Loading {} data'.format(type))
            for participant, task in tqdm(itertools.product(participants[type], range(1, 7)), total=len(participants[type] * 6)):
                session = '{}_task{}'.format(participant, task)
                df_ = pd.read_json(NBC_ROOT + 'release/{}/spatial.json'.format(session), orient='index')
                verify_spatial(df_)
                if self.args.dynamic_only:
                    df_ = df_[df_['dynamic'] == True]
                df_ = subsample(df_, self.args.subsample)
                df_['session'] = session
                self.df[type].append(df_)
            self.df[type] = pd.concat(self.df[type]).reset_index(drop=True)

        with open(tmp_path, 'wb+') as f:
            pickle.dump(self.df, f)

    def get_vgg_embeddings(self):
        vgg_embeddings = {'train': {}, 'dev': {}, 'test': {}}
        for type in participants.keys():
            for key, df in self.sequences[type].items():
                session = key[0]
                embeddings = np.load(NBC_ROOT + 'release/{}/vgg_embeddings.npz'.format(session))
                sort_indices = np.argsort(embeddings['steps'])
                vgg_steps = embeddings['steps'][sort_indices]
                vgg_z = embeddings['z'][sort_indices]
                steps = df['step'].unique()
                mask = np.isin(vgg_steps, steps)
                assert np.sum(mask) == len(steps), (vgg_steps, steps)
                vgg_embeddings[type][key] = vgg_z[mask]
        return vgg_embeddings

    def one_hot_label(self, seq, target):
        names = self.df['train']['name'].unique()
        assert target in names, (names, target)
        vocab = {v: k for k, v in enumerate(names)}
        vec = np.zeros((len(names),))
        vec[vocab[target]] = 1
        return vec

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    NBC.add_args(parser)
    args = parser.parse_args()

    nbc = NBC(args)
    #embeddings = nbc.get_vgg_embeddings()
