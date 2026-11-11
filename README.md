# REMatching

This repository contains the offical code implementation of the ACL2023 paper "RE-Matching: A Fine-Grained Semantic Matching Method for Zero-Shot Relation Extraction"

## Updated environment (2026-11-10)

The modernized code uses Python >= 3.12, Transformers 5 and Sentence
Transformers 6. Install `requirements.txt`; the original dependencies are kept
in `requirements_old.txt`. See [migration and Kaggle instructions](docs/MIGRATION.md)
for tested versions, smoke tests, and reproduction notes.

Kaggle notebook for FewRel `m = 5` and `m = 15` (five relation splits each):
[kaggle_fewrel_m5_m15.ipynb](notebooks/kaggle_fewrel_m5_m15.ipynb).
Upload it to Kaggle. Dataset discovery and the repository default training
configuration (batch size 128, five epochs, seed 42, learning rate 2e-6) are
already configured, following the `m = 10` script in `model/run_fewrel.sh`.

For separate Kaggle sessions, upload
[kaggle_fewrel_m5.ipynb](notebooks/kaggle_fewrel_m5.ipynb) and
[kaggle_fewrel_m15.ipynb](notebooks/kaggle_fewrel_m15.ipynb) as two notebooks.
Each runs five splits and writes to its own output directory. Both use the
dataset committed in this repository, so a separate Kaggle dataset attachment
is optional.

Wiki-ZSL notebooks for separate Kaggle sessions:
[m = 5](notebooks/kaggle_wikizsl_m5.ipynb),
[m = 10](notebooks/kaggle_wikizsl_m10.ipynb), and
[m = 15](notebooks/kaggle_wikizsl_m15.ipynb).
Each runs the five relation splits `40` through `44`, with batch size 128,
five epochs, training seed 42, and learning rate 2e-6. Logs, checkpoints,
per-run results and mean/sample-standard-deviation summaries are saved under
`/kaggle/working/wikizsl_m5`, `wikizsl_m10`, or `wikizsl_m15` respectively.
The notebooks use the existing FewRel notebook workflow and retain Kaggle's
CUDA-enabled Torch build.

The full Wiki-ZSL dataset is currently absent from this local checkout. Add the
processed `wikizsl_dataset.json` to `data/wikizsl/` and push it before running,
or attach it as Kaggle input. The notebooks discover the file automatically,
preferring the cloned repository. They clone the repository's current HEAD
by default so a newly pushed dataset is included, and record the actual commit
and data hash in the run manifest. Each split has five validation relations,
`m` test relations and `108 - m` training relations out of 113 total.

## Original paper environment:

> python: 3.8.3
> 
> Torch: 1.7.0

## Dependencies:

Run the following script to install dependencies.

```
pip3 install -r requirements.txt
```

## File Structure:

The file structure of this repository is shown below.

```
REMatching
├── readme.md
├── ckpt
├── data
│   ├── fewrel
│   │   ├── fewrel_dataset.json
│   │   └── relation_description
│   ├── rel2id
│   │   ├── fewrel_rel2id
│   │   └── wikizsl_rel2id
│   └── wikizsl
│       ├── wikizsl_dataset.json
│       └── relation_description
├── model
│   ├── data_process.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── train.py
│   ├── model.py
│   ├── run_fewrel.sh
│   ├── run_inference.sh
│   ├── run_wikizsl.sh
│   └── log
└── requirements.txt
```

## Datasets：

You can download the dataset used in this work via the following google drive link,and then store them in the *data/fewrel* and *data/wikizsl* paths respectively.

[FewRel (Xu et al., 2018)](https://drive.google.com/file/d/1PgSTaEEUxsE-9lhQan3Yj91pzLhxv7cT/view?usp=sharing)

[WikiZSL (Daniil Sorokin and Iryna Gurevych, 2017)](https://drive.google.com/file/d/1kGmhlpTTq8UmIUPZ2CSIruWWsi_l_ERH/view?usp=share_link)

## Train&Inference :

You can easily run training as well as inference with the following scripts

```
# training stage
# The following two script files differ only in datasets
cd model
bash run_fewrel.sh
bash run_wikizsl.sh

# testing stage
bash run_inference.sh
```

## Acknowledgement

Our implementation and experiments are based on the codes from [ZS-BERT](https://github.com/dinobby/ZS-BERT), we appreciate their open-resourced codes.

## Citation:

If the code is useful for your research project, we appreciate if you cite the following:

```
@inproceedings{zhao2023rematching,
      title={RE-Matching: A Fine-Grained Semantic Matching Method for Zero-Shot Relation Extraction}, 
      author={Jun Zhao and Wenyu Zhan and Xin Zhao and Qi Zhang and Tao Gui and Zhongyu Wei and Junzhe Wang and Minlong Peng and Mingming Sun},
      year={2023},
      booktitle = {Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      address = {Toronto, Canada},
      publisher = {Association for Computational Linguistics},
}
```
