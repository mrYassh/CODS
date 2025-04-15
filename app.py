from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, Response
from ultralytics import YOLO
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import glob
import time
import shutil
import logging
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    history = db.relationship('History', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# History model
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    original_image = db.Column(db.String(255), nullable=False)
    segmented_image = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

# Create database tables
with app.app_context():
    db.create_all()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

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

def get_latest_predict_folder():
    # Find all predict folders
    predict_folders = glob.glob('runs/segment/predict*')
    if not predict_folders:
        logger.error("No predict folders found")
        return None
    
    # Get the latest folder by number
    latest_folder = None
    latest_number = -1
    
    for folder in predict_folders:
        try:
            # Extract the number from the folder name (e.g., 'predict13' -> 13)
            folder_number = int(folder.split('predict')[-1])
            if folder_number > latest_number:
                latest_number = folder_number
                latest_folder = folder
        except ValueError:
            continue
    
    if latest_folder:
        logger.debug(f"Latest predict folder: {latest_folder}")
        return latest_folder
    else:
        logger.error("Could not find a valid prediction folder")
        return None

@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('index.html', logged_in=True)
    return render_template('index.html', logged_in=False)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!')
            return redirect(url_for('home'))
        
        flash('Invalid username or password')
        return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.')
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
@login_required

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
            
            # Perform prediction
            results = model.predict(filepath, save=True)
            logger.debug("Prediction completed")
            
            # Wait a moment for the file to be fully saved
            time.sleep(2)  # Increased wait time
            
            # Get the latest prediction folder
            latest_predict_folder = get_latest_predict_folder()
            if not latest_predict_folder:
                return jsonify({'error': 'Prediction failed - no output folder found'}), 500
            
            # Get the predicted image path
            pred_path = os.path.join(latest_predict_folder, filename)
            logger.debug(f"Looking for prediction at: {pred_path}")
            
            # List contents of the prediction folder for debugging
            try:
                folder_contents = os.listdir(latest_predict_folder)
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
                
                # Save to history
                history = History(
                    user_id=session['user_id'],
                    original_image=f'/static/uploads/{filename}',
                    segmented_image=f'/static/uploads/{output_filename}'
                )
                db.session.add(history)
                db.session.commit()
                
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

@app.route('/history')
@login_required
def history():
    user_history = History.query.filter_by(user_id=session['user_id']).order_by(History.created_at.desc()).all()
    return render_template('history.html', history=user_history)

@app.route('/live')
@login_required
def live():
    return render_template('live.html')

def generate_frames():
    camera = cv2.VideoCapture(0)  # 0 for default webcam
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Perform prediction on the frame
        results = model.predict(frame, save=False)
        
        # Get the first result
        if len(results) > 0:
            result = results[0]
            # Draw the segmentation mask on the frame
            if result.masks is not None:
                mask = result.masks.data[0].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                # Apply the mask to the frame
                frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
