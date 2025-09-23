# LaGEA

## Environment Setup

Install the conda env via:

```shell
conda create --name lagea python==3.11
conda activate lagea
pip install -r requirements.txt
pip install transformers==4.51.3 accelerate==1.10.0
pip install qwen-vl-utils[decord]
```

## Training

### Generating Expert Dataset

An optional setting in LaGEA is to use a goal image to accelerate the exploration before we collected the first successful trajectory.

```script
python main.py --config.env_name=door-open-v2-goal-hidden --config.exp_name=oracle
```

The oracle trajectory data will be saved in `data/oracle`.

### Example on Fixed-goal Task

```
python main.py --config.env_name=door-open-v2-goal-hidden --config.exp_name=lagea
```

### Example on Random-goal Task

```
python main.py --config.env_name=door-open-v2-goal-observable --config.exp_name=lagea
```

## Paper

[**LaGEA: Language Guided Embodied Agents for Robotic Manipulation**](#)

Abdul Monaf Chowdhury, AKM Moshiur Rahman Mazumder, Rabeya Akter Fariya, Safaeid Hossain
---
In submission to,
*International Conference on Learning Representations* (ICLR), 2026

<!-- ## Cite

Please cite our work if you find it useful:

```txt
@InProceedings{fu2024,
  title = {FuRL: Visual-Language Models as Fuzzy Rewards for Reinforcement Learning},
  author = {Yuwei Fu and Haichao Zhang and Di Wu and Wei Xu and Benoit Boulet},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year = {2024}
}
``` -->
