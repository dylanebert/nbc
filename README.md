# New Brown Corpus

This repository provides data and code to use the New Brown Corpus dataset. To view this dataset online, visit http://plunarlabcit.services.brown.edu/nbcapp

## Dependencies
- numpy
- pandas
- pickle
- tqdm
- argparse

## Download

- Clone this repository
- In the directory, download and extract:
  - https://plunarlabcit.services.brown.edu/release.7z
  - https://plunarlabcit.services.brown.edu/images.7z
- Set your NBC_ROOT environment variable to this directory. For example on linux, run `export NBC_ROOT=/path/to/nbc`
- If everything is set up correctly, you should be able to run nbc.py without error

## Usage

- Import the NBC class
```
import sys  
sys.path.append('path/to/nbc_parent_dir')  
from nbc.nbc import NBC
```
- Create argparse arguments to construct an NBC object.
```
import argparse
parser = argparse.ArgumentParser()
NBC.add_args(parser)
args = parser.parse_args([
  '--features', 'posX:LeftHand', 'posY:RightHand' posZ:RightHand'
])
dataset = NBC(args)
```
- Access dataset properties/methods as needed. For example, to access train/test feature matrices:
```
train_x = np.vstack(list(dataset.features['train'].values()))
test_x = np.vstack(list(dataset.features['test'].values()))
```

See nbc.py for available properties, methods, and arguments.
