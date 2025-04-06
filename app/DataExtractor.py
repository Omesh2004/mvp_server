import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
from functools import lru_cache
from xgboost import XGBClassifier
import torch
from torch import nn
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

GenreModels=[]
GenreBusy=[]

InstrumentModels=[]
InstrumentBusy=[]

GenreDict={0:"Bengali",
           1:"Bhangra",
           2:"Carnatic",
           3:"Dandiya",
           4:"Hindustani",
           5:"Kolattam",
           6:"Manipuri",
           7:"Nepali",
           8:"Rajasthani",
           9:"Uttarakhandi",
           10:"Assamese"
           }

InstrumentDict={0:"Dhol",
                1:"Flute",
                2:"Sitar",
                3:"Tabla",
                4:"Veena"}

chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

class DataExtractor:
    def __init__(self, n_mfcc=20, base_output_dir=None):
        self.n_mfcc = n_mfcc
        self.base_output_dir = base_output_dir or 'outputs'
        self.user_id = None
        self.output_dir = None
        # Store computed features to avoid recalculation
        self._feature_cache = {}

    def load_data(self, y, sr, user_id=None):
        self.y = y
        self.sr = sr
        self.user_id = user_id
        self._setup_output_dir()
        # Clear cache when loading new data
        self._feature_cache = {}
        self.feature_extract()

    def load_file(self, filename, user_id=None):
        # Use res_type='kaiser_fast' for faster loading with slight quality reduction
        self.y, self.sr = librosa.load(filename, res_type='kaiser_fast')
        self.user_id = user_id
        self._setup_output_dir()
        # Clear cache when loading new file
        self._feature_cache = {}
        self.feature_extract()
        
    def _setup_output_dir(self):
        if self.user_id:
            self.output_dir = os.path.join(self.base_output_dir, self.user_id)
            os.makedirs(self.output_dir, exist_ok=True)
        else:
            self.output_dir = self.base_output_dir

    @lru_cache(maxsize=1)
    def _get_harmonic_percussive(self):
        """Cache harmonic and percussive separation which is computationally expensive"""
        return librosa.effects.hpss(self.y)
        
    def feature_extract(self):
        features_list = {}
        
        # Compute tempo information (using hop_length for faster processing)
        tempo = librosa.beat.tempo(y=self.y, sr=self.sr, hop_length=512)[0]
        features_list['tempo'] = [tempo, tempo, tempo, 0]  # Min, mean, max, var
        
        # Get harmonic and percussive components (cached)
        self.y_harmonic, self.y_percussive = self._get_harmonic_percussive()
        
        # Calculate basic stats using numpy's vectorized operations (faster than individual calls)
        features_list['y_harmoic'] = [
            np.min(self.y_harmonic), 
            np.mean(self.y_harmonic), 
            np.max(self.y_harmonic), 
            np.var(self.y_harmonic)
        ]
        
        features_list['y_percussive'] = [
            np.min(self.y_percussive), 
            np.mean(self.y_percussive), 
            np.max(self.y_percussive), 
            np.var(self.y_percussive)
        ]
        
        # Use faster hop_length for spectral features
        hop_length = 512
        
        # Compute tonnetz only if needed
        if 'tonnetz' not in self._feature_cache:
            self._feature_cache['tonnetz'] = librosa.feature.tonnetz(
                y=self.y_harmonic, sr=self.sr, hop_length=hop_length
            )
        self.tonnetz = self._feature_cache['tonnetz']
        features_list['tonnetz'] = [
            np.min(self.tonnetz), 
            np.mean(self.tonnetz), 
            np.max(self.tonnetz), 
            np.var(self.tonnetz)
        ]

        # Compute chroma features with reduced complexity
        if 'cstft' not in self._feature_cache:
            self._feature_cache['cstft'] = librosa.feature.chroma_stft(
                y=self.y, sr=self.sr, hop_length=hop_length, n_fft=2048
            )
        cstft = self._feature_cache['cstft']
        features_list['cstft'] = [np.min(cstft), np.mean(cstft), np.max(cstft), np.var(cstft)]

        # Use vectorized operations for RMS
        if 'srms' not in self._feature_cache:
            self._feature_cache['srms'] = librosa.feature.rms(y=self.y, hop_length=hop_length)
        srms = self._feature_cache['srms']
        features_list['srms'] = [np.min(srms), np.mean(srms), np.max(srms), np.var(srms)]

        # Use a single STFT computation for multiple spectral features
        if 'stft' not in self._feature_cache:
            self._feature_cache['stft'] = np.abs(librosa.stft(self.y, hop_length=hop_length, n_fft=2048))
        
        # Reuse STFT for spectral features
        stft = self._feature_cache['stft']
        
        if 'specband' not in self._feature_cache:
            self._feature_cache['specband'] = librosa.feature.spectral_bandwidth(
                S=stft, sr=self.sr, hop_length=hop_length
            )
        specband = self._feature_cache['specband']
        features_list['specband'] = [np.min(specband), np.mean(specband), np.max(specband), np.var(specband)]

        if 'speccent' not in self._feature_cache:
            self._feature_cache['speccent'] = librosa.feature.spectral_centroid(
                S=stft, sr=self.sr, hop_length=hop_length
            )
        speccent = self._feature_cache['speccent']
        features_list['speccent'] = [np.min(speccent), np.mean(speccent), np.max(speccent), np.var(speccent)]

        if 'rolloff' not in self._feature_cache:
            self._feature_cache['rolloff'] = librosa.feature.spectral_rolloff(
                S=stft, sr=self.sr, hop_length=hop_length
            )
        rolloff = self._feature_cache['rolloff']
        features_list['rolloff'] = [np.min(rolloff), np.mean(rolloff), np.max(rolloff), np.var(rolloff)]

        # Optimize zero crossing rate calculation
        if 'zero_crossing_rate' not in self._feature_cache:
            self._feature_cache['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
                y=self.y, hop_length=hop_length
            )
        zero_crossing_rate = self._feature_cache['zero_crossing_rate']
        features_list['zero_crossing_rate'] = [
            np.min(zero_crossing_rate), 
            np.mean(zero_crossing_rate), 
            np.max(zero_crossing_rate), 
            np.var(zero_crossing_rate)
        ]

        # Compute MFCC with optimized parameters
        if 'mfcc' not in self._feature_cache:
            self._feature_cache['mfcc'] = librosa.feature.mfcc(
                y=self.y, sr=self.sr, n_mfcc=self.n_mfcc, 
                hop_length=hop_length, n_fft=2048
            )
        mfcc = self._feature_cache['mfcc']
        
        # Vectorized approach for MFCC features
        for i in range(self.n_mfcc):
            features_list[f'mfcc_{i}'] = [
                np.min(mfcc[i]), 
                np.mean(mfcc[i]), 
                np.max(mfcc[i]), 
                np.var(mfcc[i])
            ]

        # Create dataframe once at the end
        self.features_df = pd.DataFrame(features_list).transpose()
        self.features_df.columns = ['min', 'mean', 'max', 'var']

    def get_data(self, data_print=False):
        # Print the data if requested
        if data_print:
            self.print_features()

        # Vectorized reshape
        np_data = self.features_df.to_numpy().reshape((120, 1))
        return np_data

    def print_features(self):
        print(f"Tempo: {self.tempo}")
        print(self.features_df)

    def save_waveform(self, dpi=100):
        # Generate output path
        output_path = os.path.join(self.output_dir, 'waveform.png')
        
        # Plot with reduced dpi for faster saving
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(y=self.y, sr=self.sr)
        plt.title("Waveform")
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi)
        plt.close()
        
        return output_path

    def save_harmonic_percussive(self, dpi=100):
        # Generate output path
        output_path = os.path.join(self.output_dir, 'harmonic_percussive.png')
        
        # Plot with reduced dpi for faster saving
        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        librosa.display.waveshow(self.y_harmonic, sr=self.sr, color='r', alpha=0.5, ax=ax1)
        librosa.display.waveshow(self.y_percussive, sr=self.sr, color='b', alpha=0.5, ax=ax2)
        ax1.set_title("Harmonic (red) and Percussive (blue) Components")
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi)
        plt.close()
        
        return output_path

class InstrumentClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(InstrumentClassifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten_size = self._get_conv_output(input_shape)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def _get_conv_output(self, shape):
        bs = 1
        x = torch.rand(bs, 1, *shape)
        output = self.conv_layers(x)
        return int(np.prod(output.size()[1:]))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    

#it needs the  path to the audio file to be analysed
def AnalyseGenre(path):
    modelnum=-1
    for i in range(len(GenreBusy)):
        if(not GenreBusy[i]):
            modelnum=i
            break
    
    if(modelnum==-1):
        return modelnum
    
    GenreBusy[modelnum]=1

    Extractor=DataExtractor()
    Extractor.load_file(path)
    Extractor.feature_extract()
    features=Extractor.get_data()
    features=features.reshape(1,120)
    genre=GenreModels[modelnum].predict(features)
    GenreBusy[modelnum]=0
    return GenreDict[genre[0]]

def AnalyseInstrument(path):

    modelnum=-1
    for i in range(len(InstrumentBusy)):
        if(not InstrumentBusy[i]):
            modelnum=i
            break
    
    if(modelnum==-1):
        return modelnum
    
    InstrumentBusy[modelnum]=1
    
    y,sr=librosa.load(path,duration=5)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    instrumentdata = torch.tensor(np.concatenate([log_mel_spec, mfccs], axis=0), dtype=torch.float32).unsqueeze(0)

    max=0
    maxval=0
    for i,value in enumerate((InstrumentModels[modelnum])(instrumentdata).detach().cpu().numpy()[0]):
        if(value>maxval):
            maxval=value
            max=i

    return InstrumentDict[max]

def AnalyseKeyTempo(path):
    y,sr=librosa.load(path,duration=10)

    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    mean_chroma = np.mean(chromagram, axis=1)
    estimated_key_index = np.argmax(mean_chroma)
    estimated_key = chroma_to_key[estimated_key_index]

    return librosa.feature.tempo(y=y,sr=sr)[0].round(),estimated_key
    
def AnalyseMusic(path):
    genre=AnalyseGenre(path)
    instrument=AnalyseInstrument(path)
    tempo,key=AnalyseKeyTempo(path)

    return [genre,instrument,key,tempo]

def InitializeModels(num):
    global Models,Busy

    for i in range(num):
        GenreModel=XGBClassifier()  
        GenreModel.load_model("server/models/GenreModel.json")
        GenreModels.append(GenreModel)
        GenreBusy.append(0)
        
        InstrumentModel=InstrumentClassifier((141, 216),5)
        InstrumentModel.load_state_dict(torch.load("server/models/InstrumentModel.pth"))
        InstrumentModels.append(InstrumentModel)
        InstrumentBusy.append(0)


def GenerateMusic(prompt,duration,username):
    inputs = processor(
    text=prompt,
    padding=True,
    return_tensors="pt",
    )

    audio_values = model.generate(**inputs, max_new_tokens=int(256*(duration/5)))

    sampling_rate = model.config.audio_encoder.sampling_rate

    scipy.io.wavfile.write("out/{name}.mp3".format(name=username), rate=sampling_rate, data=audio_values[0, 0].numpy())