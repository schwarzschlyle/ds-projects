from flask import Flask, render_template, request
from PIL import Image
import io
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded file
    uploaded_file = request.files['file']

    # Read the image file
    image = Image.open(uploaded_file)

    # Convert the image to base64 encoded string
    buffered = io.BytesIO()
    image.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template('result.html', image_data=img_str)



@app.route('/display_image')
def display_image():
    filename = request.args.get('filename')
    response = requests.get(f'https://YOUR_REGION-YOUR_PROJECT_ID.cloudfunctions.net/display_uploaded_image?filename={filename}')
    return response.json()




if __name__ == '__main__':
    app.run(debug=True)

    