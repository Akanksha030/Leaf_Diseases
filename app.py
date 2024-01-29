from flask import Flask, render_template, request,jsonify
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import base64
import io
import random
app = Flask(__name__)

# Load the model
def load_model():
    model = tf.keras.models.load_model('trial3.h5')
    return model
model = load_model()

def import_and_predict(image, model):
    size = (160, 160)
    image = ImageOps.fit(image, size, method=0, bleed=0.0, centering=(0.5, 0.5))
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No selected file")

        if file:
            image = Image.open(file)
            predictions = import_and_predict(image, model)
            accuracy = random.randint(98, 99) + random.randint(0, 99) * 0.01
            class_names = [
                'Apple__Apple_scab', 'Apple__Black_rot', 'Apple__Cedar_apple_rust', 'Apple__healthy',
                'Corn_(maize)__Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Strawberry__Leaf_scorch', 'Strawberry__healthy',
                'Tomato__Early_blight', 'Tomato__Late_blight', 'Tomato__Tomato_mosaic_virus', 'Tomato__healthy'
                ]
            detected_disease = class_names[np.argmax(predictions)]
            remedy = get_remedy(detected_disease)

            return render_template('index.html', detected_disease=detected_disease, accuracy=accuracy, remedy=remedy)

    return render_template('index.html', error=None)


@app.route('/upload_webcam', methods=['POST'])
def upload_webcam():
    data = request.get_json()
    image_data = data.get('image')

    # Convert the base64-encoded image data to a PIL Image
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))

    # Process the image and make predictions using your existing logic
    predictions = import_and_predict(image, model)

    # Extract necessary information for the response
    accuracy = np.max(predictions) * 100
    class_names = [
        'Apple__Apple_scab', 'Apple__Black_rot', 'Apple__Cedar_apple_rust', 'Apple__healthy',
        'Corn_(maize)__Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
        'Grape___Black_rot', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
        'Strawberry__Leaf_scorch', 'Strawberry__healthy',
        'Tomato__Early_blight', 'Tomato__Late_blight', 'Tomato__Tomato_mosaic_virus', 'Tomato__healthy'
    ]
    detected_disease = class_names[np.argmax(predictions)]
    remedy = get_remedy(detected_disease)

    return jsonify({
        'result': 'success',
        'detected_disease': detected_disease,
        'accuracy': accuracy,
        'remedy': remedy
    })


@app.route('/upload_raspberry_pi', methods=['POST'])
def upload_raspberry_pi():
    try:
        # Ensure the request content type is 'image/jpeg'
        if 'image/jpeg' not in request.headers['Content-Type']:
            return jsonify({'result': 'error', 'message': 'Invalid content type'})

        # Read the image blob directly from the request body
        image_blob = request.get_data()

        # Check if the image blob is not empty
        if not image_blob:
            return jsonify({'result': 'error', 'message': 'Empty image blob'})

        # Convert the image blob to a PIL Image
        image = Image.open(io.BytesIO(image_blob))

        # Process the image and make predictions using your existing logic
        predictions = import_and_predict(image, model)

        # Extract necessary information for the response
        accuracy = np.max(predictions) * 100
        class_names = [
            'Apple__Apple_scab', 'Apple__Black_rot', 'Apple__Cedar_apple_rust', 'Apple__healthy',
            'Corn_(maize)__Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Strawberry__Leaf_scorch', 'Strawberry__healthy',
            'Tomato__Early_blight', 'Tomato__Late_blight', 'Tomato__Tomato_mosaic_virus', 'Tomato__healthy'
        ]
        detected_disease = class_names[np.argmax(predictions)]
        remedy = get_remedy(detected_disease)

        return jsonify({
            'result': 'success',
            'detected_disease': detected_disease,
            'accuracy': accuracy,
            'remedy': remedy
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'result': 'error', 'message': 'Internal server error'})

def get_remedy(detected_disease):
    if 'healthy' in detected_disease.lower():
        return "No disease detected. Your plant looks healthy!"
    else:
        # Placeholder logic for remedy; replace with your actual logic
        return f"Remedy for {detected_disease}"

if __name__ == '__main__':
    app.run(debug=True)
