# Twin Soft Actor Critic

A clean PyTorch implementation of Twin Soft Actor Critic ([SAC](https://arxiv.org/pdf/1812.05905.pdf) + [TD3](https://arxiv.org/abs/1802.09477)). The implementation is based on the [original implementation of TD3](https://github.com/sfujim/TD3).

## Usage
Experiments on single environments can be run by calling:
```bash
python main.py --env HalfCheetah-v2
```

## TODO
- [ ] Add temperature learning 
