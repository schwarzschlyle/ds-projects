from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

app = Flask(__name__)

# Load the trained model from the pickle file
pickle_path = 'banana.pickle'
with open(pickle_path, 'rb') as f:
    loaded_model = pickle.load(f)


def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Calculate the average RGB values
    average_rgb = np.mean(image, axis=(0, 1)) / 255.0  # Normalize values between 0 and 1

    # Predict the probability of ripeness for the image using the loaded model
    prediction = loaded_model.predict(average_rgb.reshape(1, -1))

    # Visualize the training process
    plt.plot(range(len(loaded_model.losses)), loaded_model.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join('static', 'training_loss_plot.png'))
    plt.close()

    return prediction


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index_2.html", error="No image file selected.")

        image_file = request.files["image"]
        image_path = os.path.join("uploads", image_file.filename)
        image_file.save(image_path)

        try:
            prediction = process_image(image_path)*100
            return render_template("index_2.html", prediction=prediction[0], plot_path="/static/training_loss_plot.png")
        except Exception as e:
            error_message = f"Error processing the image file: {str(e)}"
            return render_template("index_2.html", error=error_message)

    return render_template("index_2.html")


if __name__ == "__main__":
    app.run()
