from firebase_functions import https_fn
from firebase_admin import initialize_app
from firebase_admin import firestore, storage
from google.cloud import storage as gcs
from flask import jsonify
import tempfile
import os

# Initialize Firebase app
firebase_admin.initialize_app()

# Initialize Firestore database
db = firestore.client()

# Initialize Google Cloud Storage client
gcs_client = gcs.Client()

def process_image(event, context):
    # Get the file details from the event
    bucket = event['bucket']
    file_name = event['name']
    content_type = event['contentType']

    # Check if the file is an image
    if content_type.startswith('image/'):
        # Generate a signed URL for the uploaded image
        bucket = gcs_client.bucket(bucket)
        blob = bucket.blob(file_name)
        signed_url = blob.generate_signed_url(expiration="1 hour")

        # Save the signed URL to Firestore
        image_ref = db.collection('uploaded_images').document(file_name)
        image_ref.set({
            'filename': file_name,
            'url': signed_url
        })

# Expose the Cloud Function endpoint
def display_uploaded_image(request):
    # Get the filename parameter from the request
    filename = request.args.get('filename')

    # Fetch the image URL from Firestore
    image_ref = db.collection('uploaded_images').document(filename)
    image_data = image_ref.get().to_dict()

    if image_data:
        # Return the image URL as JSON response
        return jsonify(image_data)
    else:
        # Return an error message if the image is not found
        return jsonify({'error': 'Image not found.'})
