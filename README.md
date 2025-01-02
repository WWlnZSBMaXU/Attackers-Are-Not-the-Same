This repository contains the code for paper [Attackers Are Not the Same! Unveiling the Impact of Feature Distribution on Label Inference Attacks](https://ieeexplore.ieee.org/document/10752967) published in IEEE Transactions on Information Forensics and Security (TIFS).

# Requirements

```text
python==3.8.18
pytorch==1.11.0
scikit-learn==1.3.0
```

Or you can use `requirements.txt` and the following command to create the environment:

```bash
conda install --yes --file requirements.txt
```

# How to use?

```bash
usage: main.py [-h] [--attack {sign,cluster,reconstruction,completion}]
               [--dataset {mnist,fashionmnist,cifar10,cifar100,criteo}]
               [--epochs EPOCHS] [--batch_size BATCH_SIZE]
               [--lr_passive LR_PASSIVE] [--lr_active LR_ACTIVE]
               [--lr_attack LR_ATTACK] [--set_attack_epoch]
               [--attack_epoch ATTACK_EPOCH] [--attack_id ATTACK_ID]
               [--num_passive NUM_PASSIVE] [--use_emb]
               [--attack_every_n_iter ATTACK_EVERY_N_ITER] [--simple]
               [--padding_mode] [--division_mode {vertical,random,imbalanced}]
               [--tsne] [--as_order] [--attack_model_epochs ATTACK_MODEL_EPOCHS]
               [--lr_attack_model LR_ATTACK_MODEL] [--balanced] [--defense]
               [--defense_all] [--epsilon EPSILON] [--round ROUND]
               [--dispersion] [--our]

optional arguments:
  -h, --help            show this help message and exit
  --attack {sign,cluster,reconstruction,completion}
                        name of attack approach;
  --dataset {mnist,fashionmnist,cifar10,cifar100,criteo}
                        name of dataset;
  --epochs EPOCHS       number of epochs;
  --batch_size BATCH_SIZE
                        batch size;
  --lr_passive LR_PASSIVE
                        learning rate for passive party;
  --lr_active LR_ACTIVE
                        learning rate for active party;
  --lr_attack LR_ATTACK
                        learning rate for attacker;
  --set_attack_epoch    whether to set attack epoch;
  --attack_epoch ATTACK_EPOCH
                        epoch at which attack happens;
  --attack_id ATTACK_ID
                        id of the attacker;
  --num_passive NUM_PASSIVE
                        number of passive parties;
  --use_emb             whether to use embedding, if not, will use gradients;
  --attack_every_n_iter ATTACK_EVERY_N_ITER
                        attack every n iterations;
  --simple              use simple model
  --padding_mode        using the extreme assumption that only one passive party
                        has data and the rest are padded with random data in [0,
                        1);
  --division_mode {vertical,random,imbalanced}
                        choose the data division mode;
  --tsne                whether to use tsne for attack;
  --as_order            whether to use order dataset;
  --attack_model_epochs ATTACK_MODEL_EPOCHS
                        number of epochs for attack model (reconstruction or
                        completion model);
  --lr_attack_model LR_ATTACK_MODEL
                        learning rate for attack model (reconstruction or
                        completion model);
  --balanced            use balanced Criteo dataset;
  --defense             use defense;
  --defense_all         use defense for all passive parties;
  --epsilon EPSILON     epsilon for defense;
  --round ROUND         round for log;
  --dispersion          calculate dispersion
  --our                 use our method
```

## sign attack

Some examples:

```bash
python main.py --attack sign --dataset mnist --num_passive 2
python main.py --attack sign --dataset cifar10 --simple --num_passive 4
python main.py --attack sign --dataset mnist --num_passive 4 --division_mode random
python main.py --attack sign --dataset cifar10 --simple --num_passive 4 --division_mode imbalanced
```

Notes：

- Since the sign attack approach uses a very strong assumption of LIA (it assumes that every attacker can access to the gradient of the logit layer) and different passive parties get the same gradients, the attackers all have the same attack accuracy.

## cluster attack

Some examples:

```bash
python main.py --attack cluster --dataset mnist --num_passive 2
python main.py --attack cluster --dataset mnist --num_passive 2 --use_emb --lr_attack 0.1 --attack_id 1
python main.py --attack cluster --dataset cifar10 --simple --num_passive 4 --use_emb
python main.py --attack cluster --dataset cifar10 --simple --use_emb --num_passive 4 --division_mode imbalanced
```

## reconstruction attack

Some examples:

```bash
python main.py --attack reconstruction --dataset mnist --num_passive 2
python main.py --attack reconstruction --dataset cifar10 --simple --num_passive 2 --attack_model_epochs 100 --set_attack_epoch --attack_epoch 3
```

## completion attack

Some examples:

```bash
python main.py --attack completion --dataset fashionmnist --num_passive 2
python main.py --attack completion --dataset cifar100 --simple --num_passive 4
```

# Organization of code files

```bash
.
├── attackers
│   ├── cluster.py
│   ├── completion.py
│   ├── our.py
│   ├── reconstruction.py
│   ├── sign.py
│   └── vflbase.py
├── data         # auto create: data for reconstruction-based LIA will be auto sotred here
├── dataset      # auto create: dataset will be auto downloaded here
├── label_guess  # auto create: for model reconstruction and completion
├── log          # auto create: training, testing and attacking record will be stored here
├── main.py
├── README.md
└── utils
    ├── datasets.py
    ├── losses.py
    ├── metrics.py
    └── models.py

8 directories, 12 files
```

## main.py

The main file and program entry. Use it to load dataset, model, attacker, and to implement the LIA and our proposed defense.

## attackers/*

- **vflbase.py**: The base VFL model, all attacker classes are inherited from this class. In this file, the attack is organized as follows:
  - processing dataset
  - setup VFL model
  - setup metrics to record the information of training, testing, and attacking
  - register hook to get gradients for attacker
  - ( train or attack )
  - defense mode: *Single* and *All*
- **sign.py**: LIA using gradient sign. Due to the property of gradient sign, the attack of each passive parties is equal, because they can only use the last layer's gradient.
- **cluster.py**: LIA using cluster. You can set to use embeddings (`--use_emb`) or gradients to attack.
- **reconstruction.py**: LIA using model reconstruction. The data (embeddings, gradients, and labels) is auto stored in `data/` folder, and the surrogate labels are auto stored in `label_guess/` folder.
- **completion.py**: LIA using model completion. The data (embeddings, gradients, and labels) is auto stored in `data/` folder.
- **our.py**: Our proposed new defense strategy.

## utils/*

- **datasets.py**: Store datasets information, which can be used to load and process different datasets.
- **losses.py**: Loss classes for ExPLoit (Sanjay Kariyappa and Moinuddin K Qureshi. ExPLoit: Extracting private labels in split learning. In *SaTML*, pages 165–175, 2023.).
- **metrics.py**: Class `Metrics` to record information to `log` folder. Different datasets and attack approaches will be stored in different children folder.
- **models.py**: Neural network models for different datasets:

    | Dataset | Model |
    | --- | --- |
    | MNIST and FashionMNIST | FC1-FC1 (fully connected layer neural network) |
    | CIFAR-10/100 | Conv4-FC2 (`--simple`) and ResNet |
    | Criteo | DeepFM |

# Citation

If you use our code in your research, please cite our work：

```latex
@article{liu2025attackers,
  author  = {Liu, Yige and Wang, Che and Lou, Yiwei and Cao, Yongzhi and Wang, Hanpin},
  journal = {IEEE Transactions on Information Forensics and Security},
  title   = {Attackers Are Not the Same! Unveiling the Impact of Feature Distribution on Label Inference Attacks},
  volume  = {20},
  pages   = {71-86},
  year    = {2025},
}
```
