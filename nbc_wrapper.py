import numpy as np
import pandas as pd
from sklearn import preprocessing
import sys
sys.path.append('C:/Users/dylan/Documents')
from nbc.nbc import NBC
import argparse

PAD_VALUE = -1e9

class NBCWrapper():
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--nbc_output_type', type=str, choices=['seq2seq', 'classifier'], default='seq2seq')
        parser.add_argument('--nbc_preprocessing', nargs='+', type=str, choices=['none', 'z-norm', 'min-max', 'robust'])

    def __init__(self, args):
        self.args = args
        self.nbc = NBC(args)
        self.to_padded_dset()
        self.preprocess()

    def to_padded_dset(self):
        seq_len = 0
        for type in ['train', 'dev', 'test']:
            for seq in self.nbc.labels[type].values():
                if seq.shape[0] > seq_len:
                    seq_len = seq.shape[0]
        self.n_dim = next(iter(self.nbc.features['train'].values())).shape[-1]
        x = {}; y = {}
        for type in ['train', 'dev', 'test']:
            n = len(self.nbc.labels[type])
            x[type] = np.ones((n, seq_len, self.n_dim)).astype(np.float32) * PAD_VALUE
            if self.args.nbc_output_type == 'seq2seq':
                y[type] = np.zeros((n, seq_len)).astype(int)
            else:
                assert self.args.nbc_output_type == 'classifier'
                y[type] = np.zeros((n,)).astype(int)
            for i, (key, labels) in enumerate(self.nbc.labels[type].items()):
                feat = self.nbc.features[type][key]
                x[type][i,:feat.shape[0]] = feat
                if self.args.nbc_output_type == 'seq2seq':
                    y[type][i,:feat.shape[0]] = labels
                else:
                    assert self.args.nbc_output_type == 'classifier'
                    assert np.all(labels == labels[0]), (labels[0], labels)
                    label = labels[0]
                    y[type][i] = label
        self.x = x
        self.y = y

    def preprocess(self):
        if self.args.nbc_preprocessing is None:
            return
        for arg in self.args.nbc_preprocessing:
            x_scaled = {}
            if arg in ['z-norm', 'min-max', 'robust']:
                if arg == 'z-norm':
                    scaler = preprocessing.StandardScaler()
                elif arg == 'min-max':
                    scaler = preprocessing.MinMaxScaler()
                else:
                    assert arg == 'robust'
                    scaler = preprocessing.RobustScaler()
                mask = ~(self.x['train'] == PAD_VALUE)
                scaler.fit(self.x['train'][mask].reshape((-1, self.n_dim)))
                for type in ['train', 'dev', 'test']:
                    mask = ~(self.x[type] == PAD_VALUE)
                    x_scaled[type] = np.ones(self.x[type].shape) * PAD_VALUE
                    x_scaled[type][mask] = scaler.transform(self.x[type][mask].reshape((-1, self.n_dim))).reshape(self.x[type][mask].shape)
            self.x = x_scaled

if __name__ == '__main__':
    class Args:
        def __init__(self):
            #nbc args
            self.nbc_subsample = 9
            self.nbc_dynamic_only = True
            self.nbc_train_sequencing = 'actions'
            self.nbc_dev_sequencing = 'actions'
            self.nbc_test_sequencing = 'actions'
            self.nbc_label_method = 'hand_motion_rhand'
            self.nbc_features = ['velY:RightHand', 'relVelZ:RightHand']

            #wrapper args
            self.nbc_output_type = 'classifier'
            self.nbc_preprocessing = ['robust']
    args = Args()
    nbc_wrapper = NBCWrapper(args)
