# 2d-discharge-nn
  Neural network surrogate models of 2D discharge simulation data. Initial code written by Ikuse-san, modified by Jarl.
  
2 models:
* multilayer perceptron to predict values for each grid point in a provided grid
* 5-channel image prediction using autoencoders and an MLP to map V and P to the autoencoder latent space
