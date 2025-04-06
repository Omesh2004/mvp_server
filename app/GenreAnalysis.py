import os
import numpy as np
from xgboost import XGBClassifier
from DataExtractor import DataExtractor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GenreAnalysis')

# Global variables
Models = []
Busy = []

# Genre dictionary mapping
GenreDict = {
    0: "Bengali",
    1: "Bhangra",
    2: "Carnatic",
    3: "Dandiya",
    4: "Hindustani",
    5: "Kolattam",
    6: "Manipuri",
    7: "Nepali",
    8: "Rajasthani",
    9: "Uttarakhandi",
    10: "Assamese"
}

def AnalyseGenre(path):
    """
    Analyze the genre of an audio file at the given path.
    
    Args:
        path (str): Path to the audio file
        
    Returns:
        str or int: Genre name if successful, -1 if no models are available
    """
    # Check if models are initialized
    if len(Models) == 0:
        logger.error("Models not initialized. Initializing now...")
        InitializeModels(5)  # Initialize with 5 models
    
    modelnum = -1
    for i in range(len(Busy)):
        if not Busy[i]:
            modelnum = i
            break
    
    if modelnum == -1:
        logger.warning("No available models to process genre")
        return modelnum
    
    try:
        Busy[modelnum] = 1
        logger.info(f"Using model {modelnum} to analyze file: {path}")
        
        Extractor = DataExtractor()
        Extractor.load_file(path)
        Extractor.feature_extract()
        features = Extractor.get_data()
        features = features.reshape(1, 120)
        logger.info(f"Extracted features with shape: {features.shape}")
        
        genre = Models[modelnum].predict(features)
        logger.info(f"Predicted genre index: {genre[0]}")
        
        Busy[modelnum] = 0
        return GenreDict[genre[0]]
    
    except Exception as e:
        logger.error(f"Error in genre analysis: {str(e)}")
        Busy[modelnum] = 0  # Make sure to release the model
        raise

def InitializeModels(num):
    """
    Initialize multiple XGBoost models for genre classification.
    
    Args:
        num (int): Number of models to initialize
    """
    global Models, Busy
    
    # Clear existing models if any
    Models = []
    Busy = []
    
    logger.info(f"Initializing {num} genre classification models")
    
    # Get correct path to model file
    # Assuming the model is in the 'models' directory at the same level as 'app'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from 'app' to the server directory
    server_dir = os.path.dirname(current_dir)
    model_path = os.path.join(server_dir, "models", "GenreModel.json")
    
    logger.info(f"Looking for model at: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        # Try alternative path
        alternative_path = os.path.join(current_dir, "..", "..", "models", "GenreModel.json")
        alternative_path = os.path.abspath(alternative_path)
        logger.info(f"Trying alternative path: {alternative_path}")
        
        if os.path.exists(alternative_path):
            model_path = alternative_path
            logger.info(f"Found model at alternative path: {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at: {model_path} or {alternative_path}")
    
    for i in range(num):
        try:
            GenreModel = XGBClassifier()
            GenreModel.load_model(model_path)
            Models.append(GenreModel)
            Busy.append(0)
            logger.info(f"Model {i} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model {i}: {str(e)}")
            raise

# Initialize models when the module is imported
try:
    InitializeModels(5)
    logger.info("Models initialized successfully on module import")
except Exception as e:
    logger.error(f"Failed to initialize models on module import: {str(e)}")

