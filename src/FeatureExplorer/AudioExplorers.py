import librosa.display
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from IPython.display import Audio

class AudioGraph:

    def __init__(self):
        pass

    def view_spectrogram(self,af):
        plt.figure(figsize=(10, 4))   
        librosa.display.specshow(af.get_spectrogram(),y_axis='log', x_axis='time')
        plt.title(af.get_title_for_plot())
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()

    def view_mfcc(self,af, num_filters):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(af.get_mfcc(num_filters), x_axis='time',norm=Normalize(vmin=-30,vmax=30))
        plt.colorbar()
        plt.yticks(())
        plt.ylabel('MFC Coefficient')
        plt.title(af.get_title_for_plot())
        plt.tight_layout()

    def view_mel_spectrogram(self,af, mel_bands, max_freq):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(S=af.get_melspectrogram(mel_bands, max_freq), ref=np.mean),y_axis='mel',fmax=max_freq, x_axis='time', norm=Normalize(vmin=-20,vmax=20))
        plt.colorbar(format='%+2.0f dB',label='Amplitude')
        plt.ylabel('Mels')
        #plt.title(af.get_title_for_plot())
        plt.tight_layout()

    def view_chromagram(self,af):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(af.get_chromagram(), y_axis='chroma', x_axis='time')
        plt.colorbar(label='Relative Intensity')
        plt.title(af.get_title_for_plot())
        plt.tight_layout()

    def view_waveform(self,af):   
        plt.figure(figsize=(10,4))
        plt.subplot(1, 2, 1)
        librosa.display.waveshow(af.get_waveform(), sr=af.get_sample_rate())
        plt.title(af.get_title_for_plot())
        plt.tight_layout()  


    def view_all_grid(self,af, num_mfcc_filters, num_mel_bands, mel_fmax):
        
        fig = plt.figure(figsize=(8,8))
        axs = fig.subplot_mosaic(
            """
            AE
            BF
            C.
            D.
            """
        )

        spec =librosa.display.specshow(af.get_spectrogram(),y_axis='log', ax=axs["A"])
        axs["A"].set_title('spectrogram')
        axs["A"].tick_params(axis='y',labelsize=7)
        scb = fig.colorbar(spec, format='%+2.0f dB')
        scb.ax.tick_params(axis='y',labelsize=7)
        
        wav = librosa.display.waveshow(af.get_waveform(), sr=af.get_sample_rate(),ax=axs["E"], axis=None)
        axs["E"].set_title("wav")
        axs["E"].tick_params(axis='y',labelsize=7)

        mel = librosa.display.specshow(librosa.power_to_db(S=af.get_melspectrogram(num_mel_bands, mel_fmax), ref=np.mean),
                                    y_axis='mel',fmax=mel_fmax, norm=Normalize(vmin=-20,vmax=20),ax=axs["B"])
        axs["B"].set(title='mel spectrogram')
        axs["B"].tick_params(axis='y',labelsize=7)
        mcb = fig.colorbar(mel,format='%+2.0f dB')
        mcb.ax.tick_params(axis='y',labelsize=7)

        mfcc = librosa.display.specshow(af.get_mfcc(num_mfcc_filters),norm=Normalize(vmin=-30,vmax=30), ax=axs['C'])
        axs['C'].set(title='mfcc')
        axs['C'].tick_params(axis='y',labelsize=7)
        mfc = fig.colorbar(mfcc)
        mfc.ax.tick_params(axis='y',labelsize=7)

            
        chroma = librosa.display.specshow(af.get_chromagram(), y_axis='chroma',ax=axs['D'])
        axs['D'].set(title='chromagram')
        chr = fig.colorbar(chroma,label='Relative Intensity')
        axs['D'].tick_params(axis='y',labelsize=7)
        chr.ax.tick_params(axis='y',labelsize=7)
        
        axs['F'].set_axis_off()
        axs['F'].text(0,1,af.get_file_info(), fontsize=9)

        plt.tight_layout()

    def play_audio_file(self, af):
        y, sr = librosa.load(af.get_filepath())
        return Audio(data=y, rate=sr)