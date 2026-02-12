# TDGNet

## Environment Setup

Please refer to [inari](https://github.com/matt76k/inari.git) for environment setup.

Place TDGNet under the  [inari](https://github.com/matt76k/inari.git)  directory.

The programs in the inari and preprocess directories on this page are either not included in the original inari repository or have been modified to run TDGNet. Please place these files under the directories with the same names in the original  [inari](https://github.com/matt76k/inari.git)  repository and then execute them.

## Dataset Preparation

### Step 1: Generate a Noise-Free Dataset

First, generate a dataset without noise.  
For instructions on creating the original (noise-free) dataset, please refer to the programs in the `preprocess` directory of the [ENDNet](https://github.com/ms1211/ENDNet.git) repository.

### Step 2: Split the Dataset

After generating the dataset, split it into training, validation, and test sets by running:
```
uv run python preprocess/make_data.py
```

### Step 3: Add Noise to the Test Dataset

After the test dataset has been created, add noise by running:

```
uv run python preprocess/data_noise.py
```
