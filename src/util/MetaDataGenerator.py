import os, glob
import pandas as pd
import soundfile

class CreateRAVDESSMetadata:

    def __init__(self, folder='datasets/RAVDESS/audio/'):
        
        self.emotions = {
            '01':'neutral',
            '02':'calm',
            '03':'happy',
            '04':'sad',
            '05':'angry',
            '06':'fearful',
            '07':'disgust',
            '08':'surprised'
        }

        self.dataset_folder = folder#'../datasets/RAVDESS/audio/'


    def get_angry_label(self, filename):  
        # emotions are the 3rd part of the numerical id   
        if self.emotions[filename.split("-")[2]] == 'angry': return 1 
        else: return 0

    def get_actor(self, filename):
        return int(filename.split("-")[6].split('.')[0]) 
        
    def get_actor_sex(self,filename):
        # gender is the 7th part of the numerical id
        if int(filename.split("-")[6].split('.')[0]) % 2 == 0: return 'female'
        else: return 'male'
   
    def get_sample_rate(self, filename):
        with soundfile.SoundFile(filename) as audio:
            waveform = audio.read(dtype="float32")
            sample_rate = audio.samplerate
            return sample_rate
        

    def get_metadata(self):
        count = 0
        records = []
        for file in glob.glob(f'{self.dataset_folder}Actor_*/*.wav'):
            file = os.path.normpath(file)
            file = os.path.normpath(file)
            id = count
            filename=os.path.basename(file)

           
            filesize = os.path.getsize(file)
             
            records.append([filename, 
                           file, 
                           filesize,
                           self.get_sample_rate(file),
                           self.get_actor(filename),
                           self.get_actor_sex(filename), 
                           self.emotions[filename.split("-")[2]],
                           self.get_angry_label(filename)])
            count += 1

        # make our list of records into a dataframe with the appropriate columns names
        col_names = ['filename','filepath','filesize','sample_rate','actor','actor_sex','emotion','label']
        
        return pd.DataFrame(records, columns=col_names)