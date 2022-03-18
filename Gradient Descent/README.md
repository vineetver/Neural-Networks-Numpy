# Implementing Batch, Stochastic and Mini-Batch Gradient Descent in Numpy

Gradient descent (GD) is an iterative first-order optimization algorithm used to find a local minimum/maximum of a given function.

# 1. Basic concepts

## 1.1. Introduction

Gradient descent is an optimization algorithm that is used to find the local minimum/maximum of a given function. The algorithm is based on the concept of taking small steps in the direction of the gradient (the direction of the steepest descent) in order to find the minimum/maximum of the function. 

The gradient is a vector that contains the partial derivatives of the function with respect to each of the variables. The direction of the gradient is the direction of the steepest descent (the direction in which the function decreases the most).

## 1.2. Algorithm

The algorithm is based on the concept of taking small steps in the direction of the gradient (the direction of the steepest descent) in order to find the minimum/maximum of the function. 

The gradient is a vector that contains the partial derivatives of the function with respect to each of the variables. The direction of the gradient is the direction of the steepest descent (the direction in which the function decreases the most).

The steps of the algorithm are as follows:

1. Calculate the gradient of the function at the current point.
2. Take a small step in the direction of the gradient.
3. Repeat steps 1 and 2 until the gradient is close to zero (i.e. until the local minimum/maximum is found).

The size of the steps taken is controlled by a parameter called the learning rate. If the learning rate is too small, the algorithm will take a long time to converge. If the learning rate is too large, the algorithm might not converge.

This notebook implements Batch, Mini-Batch and Stochastic gradient descent in Numpy Python.

View code on [NBviewer](https://nbviewer.org/github/vineetver/Gradient-Descent-Numpy/blob/main/Perceptron_Gradient_Descent.ipynb)
