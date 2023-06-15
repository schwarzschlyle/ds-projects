from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import pickle
from perceptron import Perceptron
import pandas as pd



app = Flask(__name__)

# Load the model from the pickle file
with open("ao_model.pickle", "rb") as file:
    perceptron = pickle.load(file)

    
    
df_ao = pd.read_csv('./data_warehouse_proxy/apples_and_oranges.csv')
    
# Preprocess the data
X = df_ao[['Weight', 'Size']].values
y = np.where(df_ao['Class'] == 'orange', 1, -1)



# Set initial plot limits based on data range
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    
# Define the route for the home page
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        weight = float(request.form["weight"])
        size = float(request.form["size"])
        
        # Predict the class
        input_data = np.array([weight, size])
        decision_value = np.dot(input_data, perceptron.weights[1:]) + perceptron.weights[0]

        if decision_value >= 0:
            prediction = 1  # Orange
        else:
            prediction = -1  # Apple

        # Plot the data points, decision line, and input point
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='orange', label='orange')
        plt.scatter(X[y == -1, 0], X[y == -1, 1], color='green', label='apple')
        plt.scatter(input_data[0], input_data[1], color='red', label='Input Point')
        plt.xlabel('Weight')
        plt.ylabel('Size')
        plt.legend()

        # Slope-intercept form: y = mx + b
        m = -perceptron.weights[1] / perceptron.weights[2]
        b = -perceptron.weights[0] / perceptron.weights[2]

        plt.plot([x_min, x_max], [m * x_min + b, m * x_max + b], color='k', linestyle='--')

        # Set the plot limits
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        # Output the weights and biases
        weights = perceptron.weights[1:]
        bias = perceptron.weights[0]

        # Output the result
        if prediction == 1:
            result = "It's an orange."
        else:
            result = "It's an apple."

        # Save the plot with the input point
        plt.savefig("static/input_point_plot.png", bbox_inches="tight")
        plt.close()

        return render_template("index.html", result=result, weights=weights, bias=bias, plot_path="/static/input_point_plot.png")
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run()
    
    
