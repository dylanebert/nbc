import numpy as np
import pandas as pd
import itertools

participants = {
    'train': ['1_1a', '2_2a', '5_1c', '6_2c', \
        '7_1a', '8_2a', '9_1b', '10_2b', '11_1c', '12_2c', \
        '13_1a', '14_2a', '15_1b', '16_2b', '17_1c', '18_2c'],
    'test': ['3_1b' '4_2b']
}

class NBC:
    def __init__(self, subsample=90):
        self.subsample = subsample
        self.load()

    def load(self):
        for type in ['train', 'test']:
            for participant, task in itertools.product(participants[type], range(1, 7)):
                print(type, participant, task)

if __name__ == '__main__':
    print('Testing ')
    NBC()
