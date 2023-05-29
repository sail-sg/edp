# Efficient Diffusion Policy

Official Jax implementation of EDP, from the following paper: 

[Efficient Diffusion Policies for Offline Reinforcement Learning](). arxiv.  
[Bingyi Kang](), [Xiao Ma](), [Chao, Du](), [Tianyu Pang](), [Shuicheng Yan]()  
Sea AI Lab  
[[arxiv]()]

<!-- <img width="1050" alt="Screenshot 2023-03-11 at 11 02 48 AM" src="https://user-images.githubusercontent.com/17242808/224461643-c7c896ef-2ae0-4cbf-8db2-466a0e3e3576.png"> -->
---
<p align="left">
<img src="https://user-images.githubusercontent.com/17242808/224461643-c7c896ef-2ae0-4cbf-8db2-466a0e3e3576.png" width=100% height=100% 
class="left">
</p>

We propse a class of diffusion policies (EDP) that are **efficient to train** and **generally compatible to a variety of RL algorithms**.  EDP serves as a more powerful policy representation for decision making, which can be used as a plug-in replacement for feed-forward policies (**e.g.**, Gaussian policies). It has the following features: 

- [x] Enabling training diffusion with long steps, *e.g.*, 1000 steps.
- [x] $25\times$ boost in traning speed, reducing training time from 5 days to 5 hours. 
- [x] Generally applicable to both likelihood-based methods (PG, [CRR](https://arxiv.org/abs/2006.15134), [AWR](https://arxiv.org/abs/1910.00177), [IQL](https://arxiv.org/abs/2110.06169)) and value-maximization based methods (DDPG, [TD3](https://arxiv.org/pdf/1802.09477.pdf))
- [x] Setting new state-of-the-arts on all four domains in D4RL. 


## Main Results 

<p align="left">
<img src="https://user-images.githubusercontent.com/17242808/224462715-6c7a07b5-87d4-47bf-bfe2-50aa784f4275.png" width=100% height=100% 
class="left">
</p>


## Usage

Before you start, make sure to run
```bash
pip install -e .
```

Apart from this, you'll have to setup your MuJoCo environment and key as well. Please follow [D4RL](https://github.com/Farama-Foundation/D4RL) repo and setup the environment accordingly.

### Run Experiments

You can run EDP experiments using the following command:
```bash
python -m diffusion.trainer --env 'walker2d-medium-v2' --logging.output_dir './experiment_output' --algo_cfg.loss_type=TD3
```

To use other offline RL algorithms, simply change `--algo_cfg.loss_type` parameter. For example:
```bash
python -m diffusion.trainer --env 'walker2d-medium-v2' --logging.output_dir './experiment_output' --algo_cfg.loss_type=IQL --norm_reward=True
```

By default we use `ddpm` solver. To use `dpm`, set `--sample_method=dpm` and `-algo_cfg.num_timesteps=1000`.

### Weights and Biases Online Visualization Integration
This codebase can also log to [W&B online visualization platform](https://wandb.ai/site). To log to W&B, you first need to set your W&B API key environment variable.
Alternatively, you could simply run `wandb login`.

## Credits
The project structure borrows from the [Jax CQL implementation](https://github.com/young-geng/JaxCQL).

We also refer to [the diffusion model implementation from OpenAI](https://github.com/openai/guided-diffusion/tree/main/guided_diffusion) and the [official diffusion Q learning implementation](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL/).
