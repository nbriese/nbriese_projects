Variational Auto encoder
Nathan Briese 2020

- Introduction:
The goal is to implement a variational auto-encoder (VAE).
This program is inspired by Auto-Encoding Variational Bayes
Citiation: Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114v10.

- How do I plan to expand this program in the future?
I would like to get the output images to be sharper and more clear.
  One way to do this would be to use convolutional neural networks instead of fully connected ones.
  Another option might be to tune the hyperparameters using bayesian optimixation
I would like to implement CUDA support so that the model can be trained more efficiently

- How to run the program:
In addition to python3, you will need getopt, numpy, torch, torchvision, and matplotlib installed.
Run using "python3 VAE.py"
The default behavior is to generate MINST-like numbers from random noise and output those images
Use -t to train a new model
Use -l to specify the learning rate
Use -e to specify the number of epochs
