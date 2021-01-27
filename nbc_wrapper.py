import numpy as np
import pandas as pd
from sklearn import preprocessing
import sys
sys.path.append('C:/Users/dylan/Documents')
from nbc.nbc import NBC
import argparse

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
        x = {}; y = {}
        for type in ['train', 'dev', 'test']:
            x[type] = np.stack(list(self.nbc.features[type].values()), axis=0)
            if self.args.nbc_output_type == 'seq2seq':
                y[type] = np.stack(list(self.nbc.labels[type].values()), axis=0).astype(int)
            else:
                assert self.args.nbc_output_type == 'classifier'
                y[type] = np.zeros((len(self.nbc.labels[type]),)).astype(int)
                for i, labels in enumerate(self.nbc.labels[type].values()):
                    assert np.all(labels == labels[0])
                    y[type][i] = labels[0]
        self.x = x
        self.y = y

    def preprocess(self):
        if self.args.nbc_preprocessing is None:
            return
        n_dim = self.x['train'].shape[-1]
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
                scaler.fit(self.x['train'].reshape((-1, n_dim)))
                for type in ['train', 'dev', 'test']:
                    x_scaled[type] = scaler.transform(self.x[type].reshape((-1, n_dim))).reshape(self.x[type].shape)
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
