

import pandas as pd
import numpy as np
from IPython.display import Audio
import librosa.display
import matplotlib.pyplot as plt
import src.FeatureExplorer.FeatureExtractors as fe
import src.util.RecordVisualizer as rv


from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


plt.ioff()



class SampleData:
    
    def __init__(self,metadata_csv='datasets/RAVDESS/metadata/RAVDESS.csv'):
        
        self.unfiltered_df = pd.read_csv(metadata_csv,usecols=['actor','actor_sex','emotion','label','filepath'])
        self.filtered_df = self.unfiltered_df
        self.current_record_number = 0

        self.af = fe.AudioFeatures(self.filtered_df.iloc[self.current_record_number]) 

        self.manual_test_mode = 0
        self.manual_test_set = None

        self.features_train =[]
        self.features_test=[]
        self.labels_train = []
        self.labels_test=[]

        self.viz = rv.SpectroVisualizer()
        self.md_vis = rv.RecordMetaViewer()

        self.scaler = StandardScaler()


        self.mfcc_filter_count = 40
        self.mel_filter_count = 128


    def apply_actor_sex_filter(self,sex='all'):
        if sex != 'all':
            self.filtered_df = self.filtered_df[self.filtered_df['actor_sex'] == sex]
          
    
    def apply_emotion_filter(self,emotion_list):
      
        if emotion_list[0] != 'all':
            self.filtered_df = self.filtered_df[self.filtered_df.emotion.isin(emotion_list)]

    def apply_actorID_filter(self, ids):
        if ids[0] != 'all':
            actor_ids = [int(i) for i in ids]
            self.filtered_df = self.filtered_df[self.filtered_df.actor.isin(actor_ids)]

    def apply_filters(self, emotion_list, sex, ids):
        

        # start with the full set
        if self.manual_test_mode == 0:
            self.filtered_df = self.unfiltered_df.copy()
        else:
            self.filtered_df = self.manual_test_set.copy()



        self.current_record_number = 0
        
        # Filter first by emotions
        self.apply_emotion_filter(emotion_list)
       
        # Next by actor sex
        self.apply_actor_sex_filter(sex)
       
        # finally by actor id
        self.apply_actorID_filter(ids)
       
        return self.filtered_df.iloc[0:1][['actor','actor_sex','emotion','label']]


    def get_filtered_data(self):
        return self.filtered_df[['actor','actor_sex','emotion','label']]
    
    def get_unfiltered_data(self):
        return self.unfiltered_df
    
    def get_num_samples(self):
        return len(self.filtered_df)
    
    def set_to_manual_testing(self):
        self.filtered_df = self.manual_test_set
    
    def get_record_by_index(self,idx=0):
        if idx < self.get_num_samples() and idx >= 0:
            return self.filtered_df.iloc[idx:idx+1][['actor','actor_sex','emotion','label']]
    
    def get_first_record(self):
        if self.get_num_samples() > 0:
            return self.filtered_df.iloc[0:1][['actor','actor_sex','emotion','label']]
    
    def get_next_record(self):
        self.current_record_number += 1
        if self.current_record_number < self.get_num_samples():
            self.af = fe.AudioFeatures( self.filtered_df.iloc[self.current_record_number])
            return self.filtered_df.iloc[self.current_record_number:(self.current_record_number+1)][['actor','actor_sex','emotion','label']]
        
    def get_current_record_number(self):
        return self.current_record_number
    
    def get_current_record(self):
        if self.current_record_number < self.get_num_samples():
            return self.filtered_df.iloc[self.current_record_number]
        else:
            return [-1]
    
    def get_current_emotion(self):
        
        if self.current_record_number < self.get_num_samples() and self.get_num_samples() != 0:
            return self.filtered_df.iloc[self.current_record_number]['emotion']
        else:
            return "None"


    def get_current_actor_sex(self):
        if self.current_record_number < self.get_num_samples() and self.get_num_samples() != 0:
            return self.filtered_df.iloc[self.current_record_number]['actor_sex']
        else:
            return "None"
    
    def get_current_actor_id(self):
        if self.current_record_number < self.get_num_samples() and self.get_num_samples() != 0:
            return self.filtered_df.iloc[self.current_record_number]['actor']
        else:
            return "None"
    
    def get_record_audio(self):
        if self.current_record_number < self.get_num_samples() and self.get_num_samples() != 0:
            y, sr = librosa.load(self.filtered_df.iloc[self.current_record_number]['filepath'])
            return Audio(data=y, rate=sr)
        else:
            return "Sorry there's no record for those settings"
    
    def get_record_viz(self,num_mels_filters=128, mel_fmax=8000, num_mfcc_filters=40):
        if self.current_record_number < self.get_num_samples():
            self.af = fe.AudioFeatures( self.filtered_df.iloc[self.current_record_number])
        return self.viz.get_record_viz(self.af,num_mels_filters,mel_fmax,num_mfcc_filters)

    def show_record_metadata(self):
        
        return self.md_vis.show_record_metadata(self.get_current_emotion(), self.get_current_actor_sex(), self.get_current_actor_id())

    ###################################

    def choose_features(self,mfcc=40, mel=128):
        
        
        self.mfcc_filter_count = mfcc
        self.mel_filter_count = mel
        # mfcc only
        if mfcc != 'None' and mel == 'None':
            return self.scaler.fit_transform(np.load(f'datasets/RAVDESS/features/mfcc/mfcc{mfcc}.npy'))
        # mels only
        elif mfcc == 'None' and mel != 'None':
            return self.sscaler.fit_transform(np.load(f'datasets/RAVDESS/features/mel/mel{mel}.npy'))
        # both
        elif mfcc != 'None' and mel != 'None':
            mfcc_frame = np.load(f'datasets/RAVDESS/features/mfcc/mfcc{mfcc}.npy')
            mel_frame = np.load(f'datasets/RAVDESS/features/mel/mel{mel}.npy')
            feature_matrix=np.array([])
            feature_matrix = np.hstack((mfcc_frame, mel_frame))
            return self.scaler.fit_transform(feature_matrix)
        
    def random_split(self,df,test_size=0.2):
        
        dfx = df.copy()
        dfx = dfx[dfx['actor'] > 2]
        print(len(dfx))
        self.manual_test_set = df[df['actor'] <= 2].copy()
        self.manual_test_mode = 1
        labels = np.array(dfx['label'])
        features = np.vstack(dfx['features'])
        self.features_train,self.features_test,self.labels_train,self.labels_test = train_test_split(features, labels,test_size=test_size)
        