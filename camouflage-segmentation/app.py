from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import glob
import time
import shutil
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained YOLO model
model = YOLO('runs/segment/train2/weights/best.pt')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def create_new_predict_folder():
    # Create a new folder for each prediction request, use timestamp for uniqueness
    folder_name = f"predict_{int(time.time())}"
    folder_path = os.path.join('runs/segment', folder_name)
    os.makedirs(folder_path, exist_ok=True)
    logger.debug(f"Created new predict folder: {folder_path}")
    return folder_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Secure the filename and save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.debug(f"Saved uploaded file to: {filepath}")
            
            # Create a new prediction folder for this request
            prediction_folder = create_new_predict_folder()
            
            # Perform prediction and save the results to the new folder
            results = model.predict(filepath, save=True, project=prediction_folder)  # Dynamically set save path
            logger.debug("Prediction completed")
            
            # Wait a moment for the file to be fully saved
            time.sleep(2)  # Increased wait time
            
            # Get the prediction result path
            pred_path = os.path.join(prediction_folder, filename)
            logger.debug(f"Looking for prediction at: {pred_path}")
            
            # List contents of the prediction folder for debugging
            try:
                folder_contents = os.listdir(prediction_folder)
                logger.debug(f"Contents of prediction folder: {folder_contents}")
            except Exception as e:
                logger.error(f"Error listing folder contents: {str(e)}")
            
            # Move the predicted image to static folder for display
            output_filename = f'pred_{filename}'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            if os.path.exists(pred_path):
                logger.debug(f"Found prediction file at: {pred_path}")
                # Use shutil.copy2 instead of os.rename to avoid permission issues
                shutil.copy2(pred_path, output_path)
                logger.debug(f"Copied prediction to: {output_path}")
                
                # Clean up the original prediction file
                try:
                    os.remove(pred_path)
                    logger.debug("Cleaned up original prediction file")
                except Exception as e:
                    logger.error(f"Error cleaning up prediction file: {str(e)}")
                
                return jsonify({
                    'success': True,
                    'original_image': f'/static/uploads/{filename}',
                    'predicted_image': f'/static/uploads/{output_filename}'
                })
            else:
                logger.error(f"Prediction file not found at: {pred_path}")
                return jsonify({'error': 'Prediction file not found'}), 500
                
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            # Clean up any temporary files
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
