import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
import pandas as pd
import cv2

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Logistic regression model
class LogisticRegression:
    def __init__(self, lr=0.01, num_epochs=1000):
        self.lr = lr
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None
        self.losses = []

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent
        for epoch in range(self.num_epochs):
            # Linear combination
            linear_output = np.dot(X, self.weights) + self.bias

            # Activation function (sigmoid)
            y_pred = sigmoid(linear_output)

            # Calculate loss
            loss = (-1 / num_samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            self.losses.append(loss)

            # Calculate gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_output)
        return y_pred