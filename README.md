# continuous_image_adv_attack_detection
Compare efficicacy of adversarial attack detection approaches in the image domain

# Objective 

Image classification task, using VGG-16 architecture on CIFAR-100 data. Adversarially attack this model using the Projected Gradient Descent (PGD) approach. Implement _detection_ approaches to identify adversarial samples. Compare the success of detection approaches using F1 summary metrics.

# Requirements

python3.4 or above

## Necessary Packages (installation with PyPI)

pip install torch, torchvision

pip install transformers

pip install cnn_finetune

# Training

Fork the repository (and clone).

Run the _train.py_ scripts with desired arguments in your terminal. For example, to train a VGG-16 classifier:

_python ./train.py TODO_

# Experimental Results
