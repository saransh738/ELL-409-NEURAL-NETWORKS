# NEURAL-NETWORKS
## OBJECTIVE
To experiment with the use of Neural Networks for a multiclass classification problem, and try and interpret the high-level or hidden representations learnt by it. Also, to try and understand the effects of various parameter choices such as the number of hidden layers, the number of hidden neurons, and the learning rate.

## PART 1(A):

### Data
A personalised input file that contains 3000 images, each of size 28x28 pixels.Each row of this file corresponds to an image, and contains 785 comma-separated values. The first 784 are the grayscale pixel values, normalised to lie between 0 and 1 (ordering is row-major), and the last one gives the class label for the image (there are 10 classes, denoted by the labels 0 to 9).

Your task is to try and learn a Neural Network classifier for these images, starting with the raw pixels as input features, and thereby also to assess the usefulness of the different representations that your Neural Network constructs. Here is how you should proceed:

1. We would like you to do your own implementation as well as trying out a standard library for comparison.
  * Write your own implementation of a basic (fully connected) neural network for multiclass classification of the provided images: you essentially need to implement the   backpropagation algorithm to be able to train the network weights. Your implementation should allow for some flexibility (the more the better) in setting the hyperparameters and modelling choices: number of hidden layers (try with at least 1 and 2), number of units in each layer, gradient descent parameters (learning rate, batch size, convergence criterion), choice of activation function in the hidden units, mode of regularisation.

  * In addition, familiarise yourself with one Neural Network library of your choice. One suggestion is PyBrain for Python, but you can find many others. You may wish to play with a simple toy data set to get a feel for using the library, before you move on to the actual data for this assignment
    
2. Standard backpropagation neural net: Use your implementation to train a neural network to recognise the images of handwritten digits given to you. Set aside some of the data for validation, or ideally, use crossvalidation. Assess the accuracy and speed (both training and testing) of the neural net for different settings of the various hyperparameters mentioned above. Identify cases of overfitting or underfitting; use regularisation to get better results, if you think it will help. Once you have obtained a good model, try to visualise and interpret the representations being learnt by the hidden neurons. Can you make sense of them? Also, take a look at the images which are being misclassified by the network. Can you see what’s going wrong? In addition, try using the standard library, instead of your own implementation, just to train the final model with your optimised hyperparameters. Does this alter the results in any way? If so, why might that be?    
   
   
## PART 1(B):
   
Comparison with PCA features: Now consider the PCA-space representation of your data that you were provided for the previous assignment. This was a way of mapping the images to a lower-dimensional space, something that the neural net is also doing via its hidden units. Try to interpret and compare these two representations. Is the neural net in any sense able to learn a better representation than the PCA one? Train another neural network, using the PCA features from last time as the input features instead of raw pixels. First try with no hidden layers, i.e., a simple logistic regression model. Now add a hidden layer. Does it help? Why or why not? And how do these results compare with those obtained using just the raw pixels?
