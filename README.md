# DTrainer

Daniel's PyTorch trainer and dataset wrapper tools

This library provides convenient training utilities for (not complicated) PyTorch models and dataset/dataloaders. It aims to wrap around model training procedures and frees your mind to focus on what matters the most: the model itself. 

### Features
- Supports automatic tensor device conversions upon initialization. Supports cpu, CUDA, and MPS (Apple Metal) formats that PyTorch supports. AMD ROCm uses 'cuda' in PyTorch, which is theoretically also supported but not tested.
- Supports automatic saving and loading of model given a directory for model storage.
- Sets up training loop and eval loops for a variety of models. 
