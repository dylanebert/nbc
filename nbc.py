import numpy as np
import pandas as pd
import itertools
import pickle
import os
from tqdm import tqdm
import argparse
from sklearn import preprocessing
import json
import uuid
import ast

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
    if not os.path.exists(NBC_ROOT + 'tmp/cached/'):
        os.makedirs(NBC_ROOT + 'tmp/cached/')

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
    most_moving = []
    for step, group in seq.groupby('step'):
        group = group[~group['name'].isin(['Head', 'LeftHand', 'RightHand'])]
        names = group['name'].values
        speed = np.linalg.norm(group[['velX', 'velY', 'velZ']].to_numpy(), axis=1)
        most_moving.append((step, names[np.argmax(speed)]))
    return most_moving

def get_most_moving_hand(seq):
    most_moving = []
    for step, group in seq.groupby('step'):
        group = group[group['name'].isin(['LeftHand', 'RightHand'])]
        names = group['name'].values
        speed = np.linalg.norm(group[['velX', 'velY', 'velZ']].to_numpy(), axis=1)
        most_moving.append((step, names[np.argmax(speed)]))
    return most_moving

def get_feature_direct(seq, target, feat):
    rows = parse_rows(seq, target)
    return rows[feat].to_numpy()[:, np.newaxis]

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
    rows = parse_rows(seq, target)
    avg_vel = np.mean(rows[['velX', 'velY', 'velZ']].to_numpy(), axis=1)
    return reparameterize(avg_vel)

def var_vel(seq, target):
    rows = parse_rows(seq, target)
    var_vel = np.var(rows[['velX', 'velY', 'velZ']].to_numpy(), axis=1)
    return np.mean(var_vel)[np.newaxis]

def traj(seq, target):
    rows = parse_rows(seq, target)
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
    rows = parse_rows(seq, target)
    avg_pos = rows[['relPosX', 'relPosY', 'relPosZ']].to_numpy()
    return np.mean(avg_pos, axis=0)

def speed(seq, target):
    rows = parse_rows(seq, target)
    speed = rows.apply(lambda row: row['velX'] * row['velX'] + row['velY'] * row['velY'] + row['velZ'] * row['velZ'], axis=1).to_numpy()
    return speed[:, np.newaxis]

def moving(seq, target):
    rows = parse_rows(seq, target)
    speed = rows.apply(lambda row: row['velX'] * row['velX'] + row['velY'] * row['velY'] + row['velZ'] * row['velZ'], axis=1).to_numpy()
    speed[speed > 0] = 1
    return speed[:, np.newaxis]

def dist_to_rhand(seq, target):
    rows = parse_rows(seq, target)
    rhand_pos = seq[seq['name'] == 'RightHand'][['posX', 'posY', 'posZ']].to_numpy()
    target_pos = rows[['posX', 'posY', 'posZ']].to_numpy()
    dist = np.linalg.norm(target_pos - rhand_pos, axis=1)
    return dist[:, np.newaxis]

def dist_to_lhand(seq, target):
    rows = parse_rows(seq, target)
    rhand_pos = seq[seq['name'] == 'LeftHand'][['posX', 'posY', 'posZ']].to_numpy()
    target_pos = rows[['posX', 'posY', 'posZ']].to_numpy()
    dist = np.linalg.norm(target_pos - rhand_pos, axis=1)
    return dist[:, np.newaxis]

def parse_rows(seq, target):
    if isinstance(target, list):
        seq['idx'] = seq.apply(lambda row: (row['step'], row['name']), axis=1)
        seq = seq.set_index('idx', drop=True)
        rows = seq.loc[target]
    else:
        rows = seq[seq['name'] == target]
    return rows

class NBC:
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--subsample', help='subsampling step size, i.e. 1 for original 90hz, 9 for 10hz, 90 for 1hz', type=int, default=18)
        parser.add_argument('--dynamic_only', help='filter to only objects that can move', type=bool, default=True)
        parser.add_argument('--train_sequencing', choices=['token_aligned', 'chunked', 'session', 'actions'], default='session')
        parser.add_argument('--dev_sequencing', choices=['token_aligned', 'chunked', 'session', 'actions'], default='session')
        parser.add_argument('--test_sequencing', choices=['token_aligned', 'chunked', 'session', 'actions'], default='session')
        parser.add_argument('--chunk_size', help='chunk size in steps if using chunked sequencing', type=int, default=10)
        parser.add_argument('--features', nargs='+', help='feature:target, e.g. posX:Apple, dist_to_head:most_moving_obj')
        parser.add_argument('--label_method', choices=['nonzero_any', 'nonzero_by_dim', 'actions', 'actions_rhand_apple', 'pick_rhand_apple'], default='nonzero_any')
        parser.add_argument('--trim', help='idle padding to allow around actions, -1 to disable', type=int, default=-1)
        parser.add_argument('--preprocess', help='apply preprocessing to features', action='store_true')
        parser.add_argument('--recache', help='override old cached data', action='store_true')

    def __init__(self, args):
        self.args = args
        self.sequencing = {'train': self.args.train_sequencing, 'dev': self.args.dev_sequencing, 'test': self.args.test_sequencing}
        if not args.recache and self.try_load_cached():
            print('loaded cached data from args')
            return
        self.load()
        self.split_sequences()
        self.featurize()
        if args.preprocess:
            self.preprocess()
        self.generate_labels()
        self.trim()
        self.cache()

    def args_to_id(self):
        args_dict = {}
        parser = argparse.ArgumentParser()
        NBC.add_args(parser)
        args = parser.parse_args()
        for k in vars(args).keys():
            assert k in vars(self.args)
            args_dict[k] = vars(self.args)[k]
        return json.dumps(args_dict)

    def try_load_cached(self):
        args_id = self.args_to_id()
        key_path = NBC_ROOT + 'tmp/cached/keys.json'
        if not os.path.exists(key_path):
            return False
        with open(key_path) as f:
            keys = json.load(f)
        if args_id not in keys:
            return False
        fid = keys[args_id]
        fpath = NBC_ROOT + 'tmp/cached/{}.json'.format(fid)
        with open(fpath) as f:
            data = json.load(f)
        self.features = {}; self.labels = {}; self.steps = {}
        for type in ['train', 'dev', 'test']:
            self.features[type] = {}; self.labels[type] = {}; self.steps[type] = {}
            for key in data[type]['features'].keys():
                key_tuple = ast.literal_eval(key)
                self.features[type][key_tuple] = np.array(data[type]['features'][key])
                self.labels[type][key_tuple] = np.array(data[type]['labels'][key])
                self.steps[type][key_tuple] = np.array(data[type]['steps'][key])
        return True

    def cache(self):
        args_id = self.args_to_id()
        key_path = NBC_ROOT + 'tmp/cached/keys.json'
        if os.path.exists(key_path):
            with open(key_path) as f:
                keys = json.load(f)
        else:
            keys = {}
        if args_id in keys:
            return
        fid = str(uuid.uuid1())
        savepath = NBC_ROOT + 'tmp/cached/{}.json'.format(fid)
        serialized = {}
        for type in ['train', 'dev', 'test']:
            serialized[type] = {'features': {}, 'labels': {}, 'steps': {}}
            for key in self.features[type].keys():
                serialized[type]['features'][str(key)] = self.features[type][key].tolist()
                serialized[type]['labels'][str(key)] = self.labels[type][key].tolist()
                serialized[type]['steps'][str(key)] = self.steps[type][key].tolist()
        with open(savepath, 'w+') as f:
            json.dump(serialized, f)
        keys[args_id] = fid
        with open(key_path, 'w+') as f:
            json.dump(keys, f)
        print('cached data')

    def preprocess(self):
        x_train = np.vstack(list(self.features['train'].values()))
        scaler = preprocessing.MinMaxScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_dev = scaler.transform(np.vstack(list(self.features['dev'].values())))
        x_test = scaler.transform(np.vstack(list(self.features['test'].values())))
        for type in ['train', 'dev', 'test']:
            x = scaler.transform(np.vstack(list(self.features[type].values())))
            idx = 0
            for key in self.features[type].keys():
                length = self.features[type][key].shape[0]
                self.features[type][key] = np.nan_to_num(x[idx:idx+length])
                idx += length
            assert idx == x.shape[0]

    def featurize(self):
        print('featurizing')
        functions = {
            'one_hot_label': self.one_hot_label,
            'dist_to_head': dist_to_head,
            'avg_vel': avg_vel,
            'var_vel': var_vel,
            'traj': traj,
            'avg_rel': avg_rel,
            'speed': speed,
            'moving': moving,
            'dist_to_rhand': dist_to_rhand,
            'dist_to_lhand': dist_to_lhand
        }

        def parse_target(seq, target):
            if target == 'most_moving_obj':
                return [get_most_moving_obj(seq)]
            elif target == 'most_moving_hand':
                return [get_most_moving_hand(seq)]
            elif target == 'objs':
                objs1 = self.df['train'][self.df['train']['session'] == '1_1a_task1']['name'].unique()
                objs2 = self.df['train'][self.df['train']['session'] == '2_2a_task1']['name'].unique()
                objs = np.intersect1d(objs1, objs2)
                objs = objs[~np.isin(objs, ['LeftHand', 'RightHand', 'Head'])]
                return objs
            elif target == 'hands':
                return ['LeftHand', 'RightHand']
            else:
                assert target in self.df['train']['name'].unique(), target
                return [target]

        features = {'train': {}, 'dev': {}, 'test': {}}
        for type in participants.keys():
            for key, seq in tqdm(list(self.sequences[type].items())):
                n = seq['step'].unique().shape[0]
                features[type][key] = []
                if self.args.features is None:
                    features[type][key] = np.zeros((n, 1))
                    continue
                for entry in self.args.features:
                    feature, target = entry.split(':')
                    target = parse_target(seq, target)
                    feat = []
                    if feature in functions:
                        for target_ in target:
                            feat_ = functions[feature](seq, target_)
                            assert feat_.shape[0] == n or feat_.ndim == 1, (feat_.shape, n)
                            features[type][key].append(feat_)
                    else:
                        assert feature in seq.columns, feature
                        for target_ in target:
                            feat_ = get_feature_direct(seq, target_, feature)
                            assert feat_.shape[0] == n or feat_.ndim == 1, (feat_.shape, n)
                            features[type][key].append(feat_)
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
        print('generating labels')
        labels = {'train': {}, 'dev': {}, 'test': {}}
        for type in participants.keys():
            if self.args.label_method in ['actions', 'actions_rhand_apple', 'pick_rhand_apple']:
                actions = pd.read_json(NBC_ROOT + 'actions.json', orient='index')
                action_labels = ['reach', 'pick', 'put', 'retract']
                if self.args.label_method == 'actions_rhand_apple':
                    actions = actions[(actions['target'] == 'Apple') & (actions['hand'] == 'RightHand')]
                elif self.args.label_method == 'pick_rhand_apple':
                    actions = actions[(actions['target'] == 'Apple') & (actions['hand'] == 'RightHand') & (actions['action'] == 'pick')]
                    action_labels = ['pick']
                for key, steps in self.steps[type].items():
                    session = key[0]
                    group = actions[actions['session'] == session]
                    feat = self.features[type][key]
                    labels_ = np.zeros((feat.shape[0],)).astype(int)
                    actions_ = actions[actions['session'] == session]
                    for _, action in actions_.iterrows():
                        steps_ = np.arange(action['start_step'], action['end_step'])
                        for step in steps_:
                            if step in steps:
                                idx = (steps == step).argmax()
                                if labels_[idx] == 0:
                                    labels_[idx] = action_labels.index(action['action']) + 1
                        if self.sequencing[type] == 'actions':
                            if not np.all(labels_ == labels_[0]):
                                labels_[:] = 0
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
                    if self.args.trim < 0:
                        continue
                    mask = labels.copy()
                    for i in range(1, self.args.trim):
                        mask += np.roll(labels, i) + np.roll(labels, -i)
                    self.labels[type][key] = labels[mask > 0]
                    self.features[type][key] = feat[mask > 0, :]
                    self.steps[type][key] = steps[mask > 0]

    def split_sequences(self):
        print('splitting sequences')
        sequences = {'train': {}, 'dev': {}, 'test': {}}
        steps = {'train': {}, 'dev': {}, 'test': {}}
        for type in participants.keys():
            if self.sequencing[type] == 'token_aligned':
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
            elif self.sequencing[type] == 'chunked':
                for participant, task in itertools.product(participants[type], range(1, 7)):
                    session = '{}_task{}'.format(participant, task)
                    group = self.df[type][self.df[type]['session'] == session]
                    steps_ = group['step'].unique()
                    for i in range(0, steps_.shape[0] - self.args.chunk_size, self.args.chunk_size):
                        start_step, end_step = steps_[i], steps_[i + self.args.chunk_size]
                        rows = group[group['step'].isin(range(start_step, end_step))]
                        assert len(rows['step'].unique()) == self.args.chunk_size, (len(rows['step'].unique()), self.args.chunk_size)
                        sequences[type][(session, steps_[i])] = rows
                        steps[type][(session, steps_[i])] = rows['step'].unique()
            elif self.sequencing[type] == 'actions':
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
                assert self.sequencing[type] == 'session'
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
    print(next(iter(nbc.labels['test'].values())))
    print(next(iter(nbc.features['test'].values())))
    print(next(iter(nbc.features['test'].values())).shape)
    import matplotlib.pyplot as plt
    print(next(iter(nbc.steps['test'].keys())))
    plt.plot(next(iter(nbc.steps['test'].values())), next(iter(nbc.features['test'].values())))
    plt.show()
    #embeddings = nbc.get_vgg_embeddings()
