import numpy as np
import librosa.display
import soundfile
import matplotlib.pyplot as plt



class AudioFeatures:

    def __init__(self, record):
        
        self.record = record
       
        with soundfile.SoundFile(self.record.filepath) as audio:
            self.wav = audio.read(dtype="float32")
            self.sr = audio.samplerate

    
    def get_filepath(self):
        return self.record.filepath
    

    def get_spectrogram(self):
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(self.wav)), ref=np.max)
        return spectrogram

    

    def get_waveform(self):
        return self.wav
    
    def get_sample_rate(self):
        return self.sr


    # define input transformations - audio to images
    def get_chromagram(self):
        # STFT computed here explicitly; mel spectrogram and MFCC functions do this under the hood
        # Produce the chromagram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
        chromagram=librosa.feature.chroma_stft(S=np.abs(librosa.stft(self.wav)), sr=self.sr)
        return chromagram

    def get_melspectrogram(self,mel_bands,max_freq):
        # Produce the mel spectrogram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
        # Using 8khz as upper frequency bound should be enough for most speech classification tasks
        melspectrogram=librosa.feature.melspectrogram(y=self.wav, sr=self.sr, n_mels=mel_bands,fmax=max_freq)
        return melspectrogram

    def get_mel_as_numpy(self,mel_bands=128,max_freq=8000):
        # Produce the mel spectrogram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
        # Using 8khz as upper frequency bound should be enough for most speech classification tasks
        
        return np.mean(self.get_melspectrogram(mel_bands, max_freq).T, axis=0)

    def get_mfcc(self,num_filters):
        # Compute the MFCCs for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
        # 40 filterbanks = 40 coefficients
        mfc_coefficients= librosa.feature.mfcc(y=self.wav, sr=self.sr, n_mfcc=num_filters)
        return mfc_coefficients

    def get_mfcc_as_npy(self,num_filters):
        # Compute the MFCCs for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
        # 40 filterbanks = 40 coefficients
        mfc_coefficients = np.mean(self.get_mfcc(num_filters).T, axis=0)
        return mfc_coefficients


    # define the get_features function that will perform the transforms
    def get_feature_set_as_np_array(self):
        # compute features of soundfile
        chromagram = np.mean(self.feature_chromagram().T, axis=0)
        melspectrogram = np.mean(self.feature_melspectrogram(128, 8000).T, axis=0)
        mfc_coefficients = np.mean(self.feature_mfcc(40).T, axis=0)

        feature_matrix=np.array([])
        # use np.hstack to stack our feature arrays horizontally to create a feature matrix
        feature_matrix = np.hstack((chromagram, melspectrogram, mfc_coefficients))
            
        return feature_matrix               
     
