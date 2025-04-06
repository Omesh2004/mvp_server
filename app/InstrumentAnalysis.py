import os
import librosa
import numpy as np
import torch
import logging
from DataExtractor import InstrumentModels, InstrumentBusy, InstrumentDict, InstrumentClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('InstrumentAnalysis')

class InstrumentAnalyzer:
    @staticmethod
    def analyze_instrument(path):
        """
        Analyze the instrument in the given audio file, returning all probabilities.
        
        Args:
            path (str): Path to the audio file to analyze
            
        Returns:
            dict: Dictionary containing:
                - predicted_instrument: name of top predicted instrument
                - probabilities: dictionary of all instrument probabilities
                - features: dictionary of extracted features
                - status: analysis status
                or -1 if no models are available
        """
        # Check if models are initialized
        if len(InstrumentModels) == 0:
            logger.warning("Instrument models not initialized. Initializing now...")
            InstrumentAnalyzer.initialize_models(5)
        
        modelnum = -1
        for i in range(len(InstrumentBusy)):
            if not InstrumentBusy[i]:
                modelnum = i
                break
        
        if modelnum == -1:
            logger.warning("No available models to process instrument")
            return -1
        
        try:
            InstrumentBusy[modelnum] = 1
            logger.info(f"Using model {modelnum} to analyze instrument in file: {path}")
            
            # Load audio file (first 5 seconds only for efficiency)
            y, sr = librosa.load(path, duration=5)
            logger.debug(f"Audio loaded with shape: {y.shape}, sample rate: {sr}")
            
            # Extract features with fixed dimensions
            n_fft = 2048
            hop_length = 512
            
            # Mel spectrogram with fixed dimensions
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=128,
                n_fft=n_fft,
                hop_length=hop_length
            )
            log_mel_spec = librosa.power_to_db(mel_spec)
            
            # MFCCs with fixed dimensions
            mfccs = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=13,
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            # Ensure consistent shape by padding/truncating
            target_frames = 216  # Should match what the model expects
            if log_mel_spec.shape[1] < target_frames:
                # Pad with zeros
                pad_width = target_frames - log_mel_spec.shape[1]
                log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)))
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)))
            elif log_mel_spec.shape[1] > target_frames:
                # Truncate
                log_mel_spec = log_mel_spec[:, :target_frames]
                mfccs = mfccs[:, :target_frames]
            
            logger.debug(f"Processed features shapes - mel: {log_mel_spec.shape}, mfcc: {mfccs.shape}")
            
            # Prepare input tensor with correct dimensions
            combined_features = np.concatenate([log_mel_spec, mfccs], axis=0)
            instrument_data = torch.tensor(
                combined_features,
                dtype=torch.float32
            ).unsqueeze(0)  # Add batch dimension
            
            logger.debug(f"Final input tensor shape: {instrument_data.shape}")
            
            # Get model prediction probabilities for all instruments
            output = InstrumentModels[modelnum](instrument_data).detach().cpu().numpy()[0]
            probabilities = {inst: float(prob) for inst, prob in zip(InstrumentDict.values(), output)}
            
            # Get the instrument with highest probability
            max_idx = np.argmax(output)
            predicted_instrument = InstrumentDict[max_idx]
            
            # Prepare feature statistics
            feature_stats = {
                'mel_spectrogram': {
                    'mean': float(np.mean(log_mel_spec)),
                    'std': float(np.std(log_mel_spec)),
                    'min': float(np.min(log_mel_spec)),
                    'max': float(np.max(log_mel_spec))
                },
                'mfcc': {
                    'mean': float(np.mean(mfccs)),
                    'std': float(np.std(mfccs)),
                    'min': float(np.min(mfccs)),
                    'max': float(np.max(mfccs))
                }
            }
            
            result = {
                'status': 'success',
                'predicted_instrument': predicted_instrument,
                'probabilities': probabilities,
                'features': feature_stats,
                'analysis_type': 'instrument'
            }
            
            logger.info(f"Instrument analysis complete: {predicted_instrument}")
            return result
            
        except Exception as e:
            logger.error(f"Error in instrument analysis: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'analysis_type': 'instrument'
            }
            
        finally:
            if modelnum != -1:
                InstrumentBusy[modelnum] = 0
                logger.debug(f"Released model {modelnum}")

    @staticmethod
    def analyze_key_tempo(path):
        """Analyze key and tempo of audio file"""
        try:
            y, sr = librosa.load(path, duration=10)
            
            # Chroma features for key detection
            chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
            mean_chroma = np.mean(chromagram, axis=1)
            estimated_key_index = np.argmax(mean_chroma)
            estimated_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][estimated_key_index]
            
            # Tempo analysis
            tempo = float(librosa.feature.tempo(y=y, sr=sr)[0].round())
            
            return {
                'status': 'success',
                'key': estimated_key,
                'tempo': tempo,
                'analysis_type': 'key_tempo'
            }
        except Exception as e:
            logger.error(f"Error in key/tempo analysis: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'analysis_type': 'key_tempo'
            }

    @staticmethod
    def initialize_models(num_models):
        """
        Initialize instrument analysis models with correct input dimensions.
        
        Args:
            num_models (int): Number of models to initialize
            
        Raises:
            FileNotFoundError: If model file is not found
            RuntimeError: If model initialization fails
        """
        global InstrumentModels, InstrumentBusy
        
        # Clear existing models if any
        InstrumentModels = []
        InstrumentBusy = []
        
        logger.info(f"Initializing {num_models} instrument classification models")
        
        # Get correct path to model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level from 'analyzer' to 'server' directory
        server_dir = os.path.dirname(current_dir)
        model_path = os.path.join(server_dir, "models", "InstrumentModel.pth")
        
        logger.info(f"Looking for model at: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            # Try alternative path
            alternative_path = os.path.join(current_dir, "..", "..", "models", "InstrumentModel.pth")
            alternative_path = os.path.abspath(alternative_path)
            logger.info(f"Trying alternative path: {alternative_path}")
            
            if os.path.exists(alternative_path):
                model_path = alternative_path
                logger.info(f"Found model at alternative path: {model_path}")
            else:
                error_msg = f"Model file not found at: {model_path} or {alternative_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        for i in range(num_models):
            try:
                # Initialize with correct input dimensions (141, 216)
                # 141 = 128 (mel) + 13 (mfcc)
                # 216 = number of time frames
                model = InstrumentClassifier((141, 216), 5)
                model.load_state_dict(torch.load(model_path))
                model.eval()  # Set to evaluation mode
                InstrumentModels.append(model)
                InstrumentBusy.append(0)
                logger.info(f"Instrument model {i} initialized successfully")
            except Exception as e:
                error_msg = f"Failed to initialize instrument model {i}: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

# Initialize models when the module is imported
try:
    InstrumentAnalyzer.initialize_models(5)
    logger.info("Instrument models initialized successfully on module import")
except Exception as e:
    logger.error(f"Failed to initialize instrument models on module import: {str(e)}")
    raise
