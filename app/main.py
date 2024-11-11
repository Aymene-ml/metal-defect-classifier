from flask import Flask, jsonify, request, render_template
from model_utils import *  # Ensure your model loading and prediction functions are here
import io
from PIL import Image

app = Flask(__name__)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

# API endpoint for prediction
@app.route('/api/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == '':
            return jsonify({'error': 'No file uploaded'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file format'})
        
        try:
            img = Image.open(io.BytesIO(file.read()))  # Open the image from the uploaded file
            model = setup_model()  # Ensure the model setup function works
            label, prob = predict_image(model, img)  # Get prediction from the model
            data = {'label': label, 'prob': prob}  # Prepare the prediction data
            return jsonify(data)
        except Exception as e:
            return jsonify({'error': f'Error during prediction: {e}'})

# Webpage endpoint for prediction (renders HTML page)
@app.route('/predict', methods=['POST'])
def html_predict():
    file = request.files.get('file')
    if file is None or file.filename == '':
        return render_template('home.html', error="No file selected.")
    if not allowed_file(file.filename):
        return render_template('home.html', error="File format not supported.")
    
    try:
        img = Image.open(io.BytesIO(file.read()))  # Open the image from the uploaded file
        model = setup_model()  # Ensure model is set up
        label, prob = predict_image(model, img)  # Get prediction from the model
        return render_template('home.html', label=label, probability=prob)
    except Exception as e:
        return render_template('home.html', error=f"Error during prediction: {e}")

if __name__ == '__main__':
    app.run()
