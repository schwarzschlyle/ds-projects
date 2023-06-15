import numpy as np
import matplotlib.pyplot as plt
import imageio
import pickle
import os

# Define the perceptron algorithm
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)
        self.errors = []
        
        fig = plt.figure()  # Create a figure for the plot
        
        # Set initial plot limits based on data range
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        for epoch in range(1, self.epochs + 1):
            error = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                error += int(update != 0.0)
            self.errors.append(error)
            
            if epoch % 100 == 0:
                # Plot the data points and decision line
                plt.scatter(X[y == 1, 0], X[y == 1, 1], color='orange', label='orange')
                plt.scatter(X[y == -1, 0], X[y == -1, 1], color='green', label='apple')
                plt.xlabel('Weight')
                plt.ylabel('Size')
                plt.legend()

                # Slope-intercept form: y = mx + b
                m = -self.weights[1] / self.weights[2]
                b = -self.weights[0] / self.weights[2]

                plt.plot([x_min, x_max], [m * x_min + b, m * x_max + b], color='k', linestyle='--')
                
                # Set the plot limits
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                
                # Add epoch number as text on the plot
                plt.text(x_min + 0.1, y_max - 0.05 * (y_max - y_min), f"Epoch: {epoch}", color='black', fontsize=10)
                
                # Save the current plot as an image
                if not os.path.exists("plots"):
                    os.makedirs("plots")
                plt.savefig(f"plots/decision_plot_{epoch}.png")
                plt.close()  # Close the plot to clear the figure
        
        # Generate the GIF animation from the plots
        images = []
        for epoch in range(100, self.epochs + 1, 100):
            filename = f"plots/decision_plot_{epoch}.png"
            images.append(imageio.imread(filename))
        
        if not os.path.exists("plots/gif"):
            os.makedirs("plots/gif")
        imageio.mimsave("plots/gif/decision_animation.gif", images, fps=100)
        print("GIF animation saved as 'plots/gif/decision_animation.gif'.")
        
        # Save the model as a pickle file
        with open("ao_model.pickle", "wb") as file:
            pickle.dump(self, file)
        print("Model saved as 'model.pickle'.")

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]
    
    def step_function(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    def predict(self, X):
        return self.step_function(X)

    
