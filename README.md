# Harmonizing Feature Maps: A Graph Convolutional Approach for Enhancing Adversarial Robustness

Welcome to the official code repository for our paper "Harmonizing Feature Maps: A Graph Convolutional Approach for Enhancing Adversarial Robustness". This repository contains the PyTorch implementation of our proposed method.

## Implementation

The provided PyTorch code includes the implementation of our graph convolution-based denoising block, as well as the method for reconstructing the graph with correlated feature map data. We also provide example scripts for training and testing models using our proposed method.

## Directory Structure

The repository is structured as follows:

- `models/`: This directory contains the different network architectures used in our paper, including our proposed model with graph convolution-based denoising block.
- `README.md`: The file you're currently reading.
- `configs.yml`: This configuration file sets the parameters for adversarial training.
- `configs_simple.yml`: This configuration file sets the parameters for standard training.
- `configs_test.yml`: This configuration file sets the parameters for testing the model's robustness against attacks.
- `test_net.py`: This script is used for testing the standard training network's robustness against adversarial attacks.
- `train_free.py`: This script is used for adversarial training.
- `train_simple.py`: This script is used for standard training.
- `utils.py`: This file includes utility functions used across the project.
- `utils_AT.py`: This file includes utility functions specific to adversarial training.
- `validation.py`: This script is used for validating the trained models.

## Testing and Training Scripts

## Training Models

To train a network, open `train_simple.py` and set the desired network architecture. For example, you can use a wide residual network with depth 32 and widen factor 10 by setting `net = WRN32_10()`.

If you want to use our proposed method with the denoising block, you can set `net = WRN32_10_GNN(block_pos=[1,2,3])`. The `block_pos` parameter determines the positions in the network where the denoising block is added after group convolution. You can adjust these positions according to your needs.

Remember to adjust the configuration files (`configs.yml`, `configs_simple.yml`, and `configs_test.yml`) to match the settings of your training environment and the specific parameters of the models you are training.

### test_net.py

The `test_net.py` script is used to evaluate the robustness of the models trained using standard methods against adversarial attacks. The script includes four types of attacks that we utilized in our research:

- Fast Gradient Sign Method (FGSM)
- Projected Gradient Descent with 20 iterations (PGD-20)
- Projected Gradient Descent with 100 iterations (PGD-100)
- AutoAttack

These attacks provide a comprehensive evaluation of the model's robustness under different adversarial conditions.
