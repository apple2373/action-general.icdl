# Action Recognition based on Cross-Situational Action-object Statistics
```
@inproceedings{tsutsui2022action,
	author = {Satoshi Tsutsui and Xizi Wang and Guangyuan Weng and Yayun Zhang and Chen Yu and David Crandall},
	booktitle = {{International Conference on Development and Learning (ICDL)}},
	title = {Action Recognition based on Cross-Situational Action-object Statistics},
	year = {2022}
}
```

## Environment
- pytorch 1.3.1
- scikit-video 1.1.11
see `environment.yml` for deatils.
```
conda env create --prefix=conda -f environment.yml
conda activate ./conda
```

## How to run 
- set up something-something dataset on `./data/something` 
```
python main.py -g 0
```

e.g.,

```
python main.py -g 0 \
--train ./data/cvpr-v4/exp_N-375_NumCommonNouns-0_NumUniqueNouns-1_seed-0_missingexcluded_samplingseed-0.train.csv \
--val ./data/cvpr-v4/exp_N-375_NumCommonNouns-0_NumUniqueNouns-1_seed-0_missingexcluded_samplingseed-0.val.csv \
--test ./data/cvpr-v4/exp_N-375_NumCommonNouns-0_NumUniqueNouns-1_seed-0_missingexcluded_samplingseed-0.test.csv
```

## Experiment logs
- https://drive.google.com/file/d/1AM2VcSvpfS32vIn6f0lvSNQxOKSLRznB/edit
- Download it, then `tar xf experiments.tar`
- `results-main.ipynb` has the plots used in the paper.

## Note
- Currently this repository only has the 3D-ResNet18 model for RGB.
- The data splits are available in `./data/cvpr-v4`.