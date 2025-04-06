import random
import string
from bson import ObjectId
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from pymongo import ReturnDocument
from pymongo.errors import OperationFailure
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import logging
from flask_pymongo import PyMongo
import gridfs
import io
from datetime import datetime
import sys
sys.path.append('./app')
from app.MusicGenerator import generate_music

from app.GenreAnalysis import AnalyseGenre, InitializeModels
from app.InstrumentAnalysis import InstrumentAnalyzer
from app.DataExtractor import DataExtractor
#from DataExtractor import AnalyseGenre,InitializeModels
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# Connect to Musicgen database
app.config['MONGO_URI'] = "mongodb+srv://omeshmehta03:Mav6zX7W8tpVyTSo@cluster0.9xnlqg6.mongodb.net/Music?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=false&ssl=true"
#this is for production
#app.config['MONGO_URI'] = "mongodb://localhost:27017/Musicgen"  

#app.config['MONGO_URI'] = "mongodb://localhost:27017/Musicgen"
mongo = PyMongo(app)
db = mongo.db
# Define metadata collection
metadata_collection = mongo.db.selected_audios

# Set up GridFS
fs = gridfs.GridFS(mongo.db)
ALLOWED_EXTENSIONS = {'mp3'}
# Define absolute paths for better reliability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure base uploads folder exists
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Ensure outputs folder exists

# Configure CORS
cors_config = {
    "origins": ["http://localhost:3000"],
    "methods": ["GET", "POST", "DELETE","OPTIONS"],
    "allow_headers": [
        "Content-Type",
        "Authorization",
        "Accept",
        "Origin",
        "User-Agent"
    ],
    "supports_credentials": True,
    "max_age": 3600
}

# Initialize CORS with debug logging
CORS(app, **cors_config)
logger.debug("CORS initialized with config:", cors_config)

# Store processing results
processing_results = {}

@app.after_request
def after_request(response):
    logger.debug(f"After request: {response}")
    return response

@app.errorhandler(Exception)
def handle_exception(error):
    logger.error(f"Error: {str(error)}")
    return jsonify({"error": str(error)}), 500
ALLOWED_EXTENSIONS = {'mp3'}
@app.route('/')
def index():
    try:
        logger.info("Rendering test.html template")
        return render_template('test.html')
    except Exception as e:
        logger.error(f"Failed to render template: {str(e)}")
        return jsonify({
            'error': 'Template rendering failed',
            'details': str(e)
        }), 500
# Audio Upload Route
@app.route('/generate-music', methods=['POST'])
def generate_music_endpoint():
    try:
        # Get JSON data from request
        data = request.json
        
        # Extract prompt parameter
        prompt = data.get('prompt')
        
        # Validate input
        if not prompt:
            return jsonify({'error': 'Missing required prompt parameter'}), 400
        
        # Fixed duration of 10 seconds
        duration = 10
        
        # Check if output directory exists, create if it doesn't
        out_dir = os.path.join(os.path.dirname(__file__), 'out')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        # Delete existing files in the out directory
        for file in os.listdir(out_dir):
            file_path = os.path.join(out_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"Deleted existing file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        
        # Generate the music with modified function (no username)
        output_path = generate_music(prompt, duration)
        
        # Return the file path for download (now using a direct path)
        return jsonify({
            'success': True,
            'file_path': '/out/generated.mp3'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/out/<filename>', methods=['GET'])
def serve_generated_file(filename):
    """Serve generated files from the out directory"""
    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    return send_from_directory(out_dir, filename)
@app.route('/api/download/<username>', methods=['GET'])
def download_file(username):
    """Endpoint to download the generated music file"""
    try:
        # Ensure the username is safe
        safe_username = "".join(c for c in username if c.isalnum() or c in "._-")
        file_path = os.path.join('out', safe_username, 'generated.mp3')
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/upload-audio/<user_id>', methods=['POST'])
def upload_audio(user_id):
    try:
        logger.info(f"New audio upload request received for user: {user_id}")
        
        # Validate file presence
        if 'audio' not in request.files:
            logger.warning("No audio file part in request")
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            logger.warning("Empty filename provided")
            return jsonify({'error': 'No selected file'}), 400
        
        # Validate file type
        if not audio_file.content_type.startswith('audio/'):
            logger.warning(f"Invalid file type: {audio_file.content_type}")
            return jsonify({'error': 'File must be an audio file'}), 400
        
        # Validate file size (50MB limit)
        if audio_file.content_length > 50 * 1024 * 1024:
            return jsonify({'error': 'File size exceeds 50MB limit'}), 400
        
        # Get metadata from form
        description = request.form.get('description', '')
        tags = request.form.get('tags', '').split(',') if request.form.get('tags') else []
        
        # Create metadata document
        metadata = {
            'user_id': user_id,
            'filename': secure_filename(audio_file.filename),
            'original_filename': audio_file.filename,
            'content_type': audio_file.content_type,
            'description': description,
            'tags': tags,
            'uploaded_at': datetime.utcnow()
        }
        
        # Store file in GridFS
        audio_file.seek(0)  # Reset file pointer to beginning
        gridfs_file_id = fs.put(
            audio_file,
            filename=secure_filename(audio_file.filename),
            content_type=audio_file.content_type,
            metadata=metadata
        )
        logger.info(f"Audio saved to GridFS with ID: {gridfs_file_id}")
        
        # Return success response
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'filename': secure_filename(audio_file.filename),
            'file_id': str(gridfs_file_id)
        }), 200
        
    except Exception as e:
        logger.error(f"Error uploading audio: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/audio-files/<user_id>', methods=['GET'])
def get_user_audio_files(user_id):
    try:
        # Get all audio files for the user from metadata collection
        files = list(metadata_collection.find({'user_id': user_id}))
        
        # Convert ObjectId to string for JSON serialization
        for file in files:
            file['_id'] = str(file['_id'])
            file['fileId'] = str(file['fileId'])
        
        return jsonify({
            'success': True,
            'files': files
        }), 200
    
    except Exception as e:
        logger.error(f"Error getting audio files: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500
@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        logger.info("Processing audio from upload folder")
        
        # Check upload folder
        upload_files = os.listdir(UPLOAD_FOLDER)
        if not upload_files:
            logger.error("No files found in upload folder")
            return jsonify({'error': 'No audio files found in upload folder'}), 404
        
        # Clean output folder first
        for existing_file in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, existing_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    logger.info(f"Cleared existing output file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing file {file_path}: {str(e)}")
                return jsonify({
                    'error': 'Failed to clear output folder',
                    'details': str(e)
                }), 500
        
        # Process the audio file
        filename = upload_files[0]  # Get first/only file
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        logger.info(f"Processing file: {file_path}")
        
        data_extractor = DataExtractor(base_output_dir=OUTPUT_FOLDER)
        data_extractor.load_file(file_path)
        
        # Generate visualizations
        waveform_path = data_extractor.save_waveform()
        harmonic_path = data_extractor.save_harmonic_percussive()
        
        # Get output filenames
        waveform_filename = os.path.basename(waveform_path)
        harmonic_filename = os.path.basename(harmonic_path)
        
        # Analyze genre
        genre = AnalyseGenre(file_path)
        if genre == -1:
            logger.warning("No available models. Attempting to reinitialize...")
            InitializeModels(5)
            genre = AnalyseGenre(file_path)
            if genre == -1:
                return jsonify({'error': 'No available models to process genre'}), 503

        return jsonify({
            'status': 'success',
            'waveform_url': f"/outputs/{waveform_filename}",
            'harmonic_url': f"/outputs/{harmonic_filename}",
            'genre': genre
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': 'Internal server error',
            'details': str(e)
        }), 500
@app.route('/analyze-music', methods=['POST'])
def analyze_music():
    """
    Comprehensive music analysis for the latest audio file in uploads folder.
    
    Returns:
        JSON response with:
        - genre analysis
        - instrument analysis (with probabilities)
        - key/tempo analysis
        - audio features
    """
    try:
        logger.info("Starting music analysis for uploaded file")
        
        # Check upload folder
        upload_files = os.listdir(UPLOAD_FOLDER)
        if not upload_files:
            logger.error("No files found in upload folder")
            return jsonify({
                'status': 'error',
                'error': 'No audio files found',
                'details': 'Upload folder is empty'
            }), 404
        
        # Get the most recent file
        latest_file = max(
            [os.path.join(UPLOAD_FOLDER, f) for f in upload_files],
            key=os.path.getmtime
        )
        
        # Perform all analyses
        from InstrumentAnalysis import InstrumentAnalyzer
        from GenreAnalysis import AnalyseGenre
        
        genre = AnalyseGenre(latest_file)
        instrument_analysis = InstrumentAnalyzer.analyze_instrument(latest_file)
        key_tempo_analysis = InstrumentAnalyzer.analyze_key_tempo(latest_file)
        
        # Compile final results
        result = {
            'status': 'success',
            'filename': os.path.basename(latest_file),
            'analyses': {
                'genre': genre if isinstance(genre, str) else 'Unknown',
                'instrument': instrument_analysis,
                'key_tempo': key_tempo_analysis
            }
        }
        
        logger.info("Music analysis complete")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error analyzing music: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/check_user', methods=['POST'])
def check_user():
    try:
        data = request.json
        user_id = data.get('id')

        if not user_id:
            return jsonify({"error": "No user ID provided"}), 400

        # Check if user exists by ID
        existing_user = mongo.db.users.find_one({"id": user_id})
        
        return jsonify({
            "exists": existing_user is not None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/random-audio', methods=['GET'])
def get_random_audio():
    try:
        # Specific user ID to fetch audio from
        user_id = "user_2tAWzAngClCUsUP1mB61AP12tjV"
        
        # Find the specific user's document
        user = db.users.find_one({"id": user_id})
        
        if not user:
            return jsonify({"error": "User not found"}), 404
            
        if 'audio_files' not in user or not user['audio_files']:
            return jsonify({"error": "No audio files found for this user"}), 404
        
        # Select a random audio file from this user's files
        random_audio = random.choice(user['audio_files'])
        
        # Get the file from GridFS
        gridfs_id = random_audio['gridfs_id']
        audio_file = fs.get(ObjectId(gridfs_id))
        
        # Send the file with appropriate content type
        return send_file(
            audio_file,
            mimetype=random_audio.get('content_type', 'audio/mpeg'),
            as_attachment=False,
            download_name=random_audio.get('filename', 'audio.mp3')
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/uploadnew', methods=['POST'])
def upload_file():
    try:
        # Validate file presence
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
            
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
            
        if file:
            # Ensure upload folder exists
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Remove all existing files in the upload folder
            for existing_file in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, existing_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    return jsonify({
                        'error': 'Failed to remove existing files',
                        'details': str(e)
                    }), 500
            
            # Secure the filename and create path
            filename = secure_filename(file.filename)
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save the file
            file.save(upload_path)
            
            return jsonify({
                'message': 'File uploaded successfully',
                'filename': filename,
                'filepath': upload_path
            }), 200
            
    except Exception as e:
        return jsonify({
            'error': 'Upload failed',
            'details': str(e)
        }), 500
@app.route('/upload-edit/<user_id>/<role>', methods=['POST'])
def upload_to_gridfs(user_id, role):
    try:
        logger.info(f"New upload request received for user: {user_id}, role: {role}")
        
        # Validate request method
        if request.method != 'POST':
            return jsonify({"message": "Method not allowed"}), 405
        
        # Validate file presence
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.warning("Empty filename provided")
            return jsonify({'error': 'No selected file'}), 400
            
        if file:
            filename = secure_filename(file.filename)
            logger.info(f"Processing file: {filename}")
            
            # Save directly to GridFS with enhanced metadata
            content_type = file.content_type if hasattr(file, 'content_type') else 'application/octet-stream'
            gridfs_file_id = fs.put(
                file,
                filename=filename,
                content_type=content_type,
                metadata={
                    'user_id': user_id,
                    'role': role,
                    'original_filename': filename,
                    'content_type': content_type,
                    'uploaded_at': datetime.utcnow()
                }
            )
            logger.info(f"File saved to GridFS with ID: {gridfs_file_id}")
            
            # Update user document with new audio file reference
            result = mongo.db.users.update_one(
                {'id': user_id},
                {'$addToSet': {
                    'audio_files': {
                        'gridfs_id': gridfs_file_id,
                        'filename': filename,
                        'content_type': content_type,
                        'role': role,
                        'uploaded_at': datetime.utcnow()
                    }
                }}
            )
            
            if result.modified_count == 0:
                # If user doesn't exist, create a new user document
                mongo.db.users.insert_one({
                    'id': user_id,
                    'audio_files': [{
                        'gridfs_id': gridfs_file_id,
                        'filename': filename,
                        'content_type': content_type,
                        'role': role,
                        'uploaded_at': datetime.utcnow()
                    }]
                })
                logger.info(f"Created new user document for ID: {user_id}")
            
            return jsonify({
                'message': f'File {filename} uploaded successfully',
                'filename': filename,
                'gridfs_id': str(gridfs_file_id),
                'role': role
            }), 200
            
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return jsonify({
            'error': 'Upload failed',
            'details': str(e)
        }), 500
@app.route('/outputs/<path:path>')
def serve_output(path):
    """Serve static files from the outputs directory"""
    try:
        logger.info(f"Serving output file: {path}")
        return send_from_directory(OUTPUT_FOLDER, path)
    except Exception as e:
        logger.error(f"Error serving output file: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

@app.route('/files', methods=['GET'])
def get_user_files():
    try:
        user_id = request.args.get('userId')
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        user = mongo.db.users.find_one({'id': user_id})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        audio_files = user.get('audio_files', [])
        return jsonify(audio_files)
    except Exception as e:
        logger.error(f"Error fetching user files: {str(e)}")
        return jsonify({'error': 'Failed to fetch files'}), 500
@app.route('/files/<file_id>', methods=['DELETE'])
def delete_file(file_id):
    try:
        logger.info(f"Deleting file with ID: {file_id}")
        user_id = request.args.get('userId')
        
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        # Convert string ID to ObjectId
        try:
            file_id_obj = ObjectId(file_id)
        except:
            logger.warning(f"Invalid file ID format: {file_id}")
            return jsonify({'error': 'Invalid file ID format'}), 400
        
        # Find the user
        user = mongo.db.users.find_one({'id': user_id})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if the file exists and belongs to the user
        file_exists = False
        for file in user.get('audio_files', []):
            if str(file.get('gridfs_id')) == str(file_id_obj):
                file_exists = True
                break
        
        if not file_exists:
            return jsonify({'error': 'File not found or does not belong to user'}), 404
        
        # Delete file from GridFS
        if fs.exists({"_id": file_id_obj}):
            fs.delete(file_id_obj)
        
        # Remove file reference from user document
        result = mongo.db.users.update_one(
            {'id': user_id},
            {'$pull': {'audio_files': {'gridfs_id': file_id_obj}}}
        )
        
        if result.modified_count == 0:
            return jsonify({'error': 'Failed to update user document'}), 500
        
        return jsonify({'message': 'File deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"File deletion failed: {str(e)}")
        return jsonify({
            'error': 'File deletion failed',
            'details': str(e)
        }), 500
@app.route('/test-json', methods=['POST'])
def test_json():
    try:
        logger.info("New JSON test request received")
        data = request.get_json()
        
        if not data:
            logger.warning("No JSON data provided")
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Insert data into music collection
        result = mongo.db.music.insert_one(data)
        
        logger.info(f"JSON data inserted successfully: {result.inserted_id}")
        return jsonify({
            'message': 'JSON data inserted successfully',
            'inserted_id': str(result.inserted_id),
            'data': data
        }), 200
        
    except Exception as e:
        logger.error(f"JSON test failed: {str(e)}")
        return jsonify({
            'error': 'JSON test failed',
            'details': str(e)
        }), 500
@app.route('/generate-save/<user_id>', methods=['POST'])
def save_audio(user_id):
    try:
        logger.info(f"Received save request for user {user_id}")
        
        # Generate a unique filename using timestamp and a random string
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        unique_filename = f"audio_{timestamp}_{random_suffix}.mp3"
        
        # Source filename is still generated.mp3
        source_filename = "generated.mp3"
        
        # Check if user exists
        user = mongo.db.users.find_one({"id": user_id})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Path to the user-specific subfolder containing generated.mp3
        out_folder = os.path.join(os.path.dirname(__file__), 'out')
        user_folder = os.path.join(out_folder, f"{user_id}")
        file_path = os.path.join(user_folder, source_filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({'error': f'Generated audio file not found in {user_folder}'}), 404
        
        # Read the file
        with open(file_path, 'rb') as audio_file:
            # Save to GridFS with the unique filename
            audio_id = fs.put(
                audio_file,
                filename=unique_filename,
                content_type='audio/mpeg'  # MP3 MIME type
            )
            
            # Update user document in generated-audio array
            mongo.db.users.update_one(
                {"id": user_id},
                {
                    "$push": {
                        "generated-audio": {
                            "gridfs_id": audio_id,
                            "filename": unique_filename,
                            "created_at": datetime.utcnow(),
                            "content_type": "audio/mpeg"
                        }
                    }
                }
            )
        
        logger.info(f"Successfully saved MP3 file {unique_filename} for user {user_id}")
        return jsonify({
            'success': True,
            'message': 'MP3 audio saved successfully',
            'filename': unique_filename
        })
        
    except Exception as e:
        logger.error(f"Failed to save audio: {str(e)}")
        return jsonify({'error': str(e)}), 500
# Route to add users to the users collection
@app.route('/user', methods=['POST'])  # Fixed missing @ decorator
def add_user():
    try:
        logger.info("New user registration request received")
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        required_fields = ['fullName', 'email', 'id']
        if any(field not in data or not data[field] for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        user_id = data['id']
        existing_user = mongo.db.users.find_one({"id": user_id})
        
        if existing_user:
            # Check if any audio files exist in GridFS
            if 'audio_files' in existing_user:
                for audio_file in existing_user['audio_files']:
                    try:
                        # Try to delete from GridFS
                        fs.delete(audio_file['gridfs_id'])
                        logger.info(f"Deleted audio file {audio_file['filename']} from GridFS")
                    except Exception as e:
                        logger.error(f"Failed to delete audio file {audio_file['filename']}: {str(e)}")
            
            return jsonify({
                'message': 'User already exists',
                'user': existing_user
            }), 200
        
        # Create new user
        user_data = {
            'fullName': data['fullName'],
            'email': data['email'],
            'id': user_id,
            'audio_files': [],
            'generated-audio': [] # Initialize empty array for future audio files
        }
        
        result = mongo.db.users.insert_one(user_data)
        return jsonify({
            'message': 'User added successfully',
            'inserted_id': str(result.inserted_id),
            'user': user_data
        }), 201
        
    except Exception as e:
        logger.error(f"User registration failed: {str(e)}")
        return jsonify({
            'error': 'User registration failed',
            'details': str(e)
        }), 500
@app.route('/files-generated', methods=['GET'])
def get_user_generated_files():
    try:
        user_id = request.args.get('userId')
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        user = mongo.db.users.find_one({'id': user_id})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        generated_audio_files = user.get('generated-audio', [])
        return jsonify(generated_audio_files)
    except Exception as e:
        logger.error(f"Error fetching user generated files: {str(e)}")
        return jsonify({'error': 'Failed to fetch generated files'}), 500

@app.route('/files-generated/<file_id>', methods=['DELETE'])
def delete_generated_file(file_id):
    try:
        logger.info(f"Deleting generated file with ID: {file_id}")
        user_id = request.args.get('userId')
        
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400
        
        # Convert string ID to ObjectId
        try:
            file_id_obj = ObjectId(file_id)
        except:
            logger.warning(f"Invalid file ID format: {file_id}")
            return jsonify({'error': 'Invalid file ID format'}), 400
        
        # Find the user
        user = mongo.db.users.find_one({'id': user_id})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if the file exists and belongs to the user
        file_exists = False
        for file in user.get('generated-audio', []):
            if str(file.get('gridfs_id')) == str(file_id_obj):
                file_exists = True
                break
        
        if not file_exists:
            return jsonify({'error': 'Generated file not found or does not belong to user'}), 404
        
        # Delete file from GridFS
        if fs.exists({"_id": file_id_obj}):
            fs.delete(file_id_obj)
        
        # Remove file reference from user document
        result = mongo.db.users.update_one(
            {'id': user_id},
            {'$pull': {'generated-audio': {'gridfs_id': file_id_obj}}}
        )
        
        if result.modified_count == 0:
            return jsonify({'error': 'Failed to update user document'}), 500
        
        return jsonify({'message': 'Generated file deleted successfully'}), 200
        
    except Exception as e:
        logger.error(f"Generated file deletion failed: {str(e)}")
        return jsonify({
            'error': 'Generated file deletion failed',
            'details': str(e)
        }), 500

@app.route('/files-generated/<file_id>', methods=['GET'])
def get_generated_file(file_id):
    try:
        logger.info(f"Retrieving generated file with ID: {file_id}")
        
        # Convert string ID to ObjectId
        try:
            file_id = ObjectId(file_id)
        except:
            logger.warning(f"Invalid file ID format: {file_id}")
            return jsonify({'error': 'Invalid file ID format'}), 400
        
        # Check if the file exists in GridFS
        if not fs.exists({"_id": file_id}):
            logger.warning(f"Generated file with ID {file_id} not found in GridFS")
            return jsonify({'error': 'Generated file not found'}), 404
        
        grid_out = fs.get(file_id)
        content_type = grid_out.content_type
        filename = grid_out.filename
        
        return send_file(
            io.BytesIO(grid_out.read()),
            mimetype=content_type,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Generated file retrieval failed: {str(e)}")
        return jsonify({
            'error': 'Generated file retrieval failed',
            'details': str(e)
        }), 500
# Add a route to retrieve files from GridFS
@app.route('/files/<file_id>', methods=['GET'])
def get_file(file_id):
    try:
        logger.info(f"Retrieving file with ID: {file_id}")
        
        # Convert string ID to ObjectId
        try:
            file_id = ObjectId(file_id)
        except:
            logger.warning(f"Invalid file ID format: {file_id}")
            return jsonify({'error': 'Invalid file ID format'}), 400
        
        # Check if the file exists in GridFS
        if not fs.exists({"_id": file_id}):
            logger.warning(f"File with ID {file_id} not found in GridFS")
            return jsonify({'error': 'File not found'}), 404
        
        grid_out = fs.get(file_id)
        content_type = grid_out.content_type
        filename = grid_out.filename
        
        return send_file(
            io.BytesIO(grid_out.read()),
            mimetype=content_type,
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        logger.error(f"File retrieval failed: {str(e)}")
        return jsonify({
            'error': 'File retrieval failed',
            'details': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)