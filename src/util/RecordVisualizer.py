import librosa.display
import matplotlib.pyplot as plt
import src.FeatureExplorer.FeatureExtractors as fe
from matplotlib.colors import Normalize
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay,DetCurveDisplay
import numpy as np

plt.ioff()

class SpectroVisualizer:

    def __init__(self):
        self.fig = plt.figure(figsize=(5,7), layout="constrained")
        
        
    def get_record_viz(self,af,num_mels_filters=128, mel_fmax=8000, num_mfcc_filters=40,):
        
        self.fig.clf()

        self.axs = self.fig.subplot_mosaic(
            """
            AAAA
            BBBB
            CCCC
            """)

        mel = librosa.display.specshow(librosa.power_to_db(af.get_melspectrogram(num_mels_filters, mel_fmax), ref=np.mean),
                                                        y_axis='mel',fmax=mel_fmax, norm=Normalize(vmin=-20,vmax=20), ax=self.axs["A"])
        
        self.axs["A"].set(title='mel spectrogram')
        self.axs["A"].tick_params(axis='y',labelsize=7)
        mcb = self.fig.colorbar(mel,format='%+2.0f dB')
        mcb.ax.tick_params(axis='y',labelsize=7)

        mfcc = librosa.display.specshow(af.get_mfcc(num_mfcc_filters),norm=Normalize(vmin=-30,vmax=30), ax=self.axs['B'])
        self.axs['B'].set(title='mfcc')
        self.axs['B'].tick_params(axis='y',labelsize=7)
        mfc = self.fig.colorbar(mfcc)
        mfc.ax.tick_params(axis='y',labelsize=7)

        wav = librosa.display.waveshow(af.get_waveform(), sr=af.get_sample_rate(),axis='time', ax=self.axs['C'])
        self.axs['C'].set(title='wave')
        self.axs['C'].tick_params(axis='y',labelsize=7)

        self.fig.canvas.header_visible = False
        
        return self.fig
    
class RecordMetaViewer:

    def __init__(self):
        

        self.fig = plt.figure(figsize=(2,1))

    def show_record_metadata(self,emotion, sex, id):
        
        self.fig.clf()
        axs = self.fig.subplots()
        self.fig.set_size_inches(2.2, 1)
        cols = ['emotion','actor_sex','actor_id']
        cells = [emotion, sex, str(id)]

        self.fig.canvas.header_visible = False
        self.fig.patch.set_visible(False)
        axs.axis('off')
        axs.axis('tight')
        self.fig.tight_layout()
        table = axs.table( colLabels=cols,cellText=[cells],loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.5,1.5)
        #plt.close()
        return self.fig
    

class ViewGridSearchResults:
    
    def __init__(self):
        self.fig  = plt.figure(figsize=(4,3))

    def print_records_to_table(self, records):
        
        self.fig.clf()
        
        cols = records[0]
        cell_text = records[1]

        ax = self.fig.subplots()
        self.fig.suptitle("Top Grid Search Results")
        self.fig.set_size_inches(4, 3)
        self.fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        self.fig.tight_layout()
        self.fig.canvas.header_visible = False
        
        col_widths = [0.15,0.16,0.17,0.16,0.18,0.18]

        table = ax.table(cellText=cell_text,colLabels=cols,loc='center',colWidths=col_widths)
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0,1.5)
       
        return self.fig


class ViewPrecisionRecall:
    

    def __init__(self):

        self.fig = plt.figure(figsize=(4,3))
       
        
    def show_precision_recall(self,train_model,features_train,features_test,labels_train,labels_test,test_model,test_mode):
        self.fig.clf()
        
        ax = self.fig.subplots()
        self.fig.set_size_inches(4,3)
        self.fig.canvas.header_visible = False
        ax.set_title("Precision Recall Curve (PRC)")
        PrecisionRecallDisplay.from_estimator(train_model, features_train, labels_train,ax=ax,name='train',plot_chance_level=True) 
        
        if test_mode == 1:
            pred = test_model.decision_function(features_test)
            PrecisionRecallDisplay.from_predictions(labels_test,pred,ax=ax,name="test",plot_chance_level=True)
        ax.set_aspect('auto')
        #ax.set_box_aspect(0.7)
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.xaxis.set_label_coords(0.6, -.15)
        
        ax.legend(bbox_to_anchor=(-0.1, -0.13), loc='upper left', borderaxespad=0.1)

        self.fig.tight_layout()
        return self.fig



class ViewConfusionMatrix:
    
    def __init__(self):
        self.train_fig = plt.figure(figsize=(4,3))
        self.test_fig = plt.figure(figsize=(4,3))

    def show_confusion_matrix_train(self,model,features_train,labels_train):
        
        self.train_fig.clf()
        self.train_fig.set_size_inches(3,3)
        axs = self.train_fig.subplots()
        axs.set_title("Confusion Matrix Train",fontsize=12)
        
        ConfusionMatrixDisplay.from_estimator(model, features_train, labels_train,ax=axs,colorbar=False)
        self.train_fig.tight_layout()
        return self.train_fig
    
    def show_confusion_matrix_test(self,test_mode, labels_test,predictions):
        self.test_fig.clf()
        self.test_fig.set_size_inches(3,3)
        axs = self.test_fig.subplots()
        axs.set_title("Confusion Matrix Test",fontsize=12)
        if test_mode == 1:
            ConfusionMatrixDisplay.from_predictions(labels_test,predictions,ax=axs,colorbar=False,cmap="magma")
        self.test_fig.tight_layout()
        return self.test_fig


class ViewModelMetrics:
    
    def __init__(self):

        self.fig = plt.figure(figsize=(2,1.25))

    def show_train_metrics(self,train_record):
        
        self.fig.clf()

        cols = train_record[0]
        cell_text = train_record[1]

        self.fig.set_size_inches(2,1.25)
        rows = ["Train Results"]
        axs = self.fig.subplots()
        self.fig.suptitle("Model Performance")
        self.fig.canvas.header_visible = False
        self.fig.patch.set_visible(False)
        axs.axis('off')
        axs.axis('tight')
        self.fig.tight_layout()
        table = axs.table( colLabels=cols,rowLabels=rows,cellText=[cell_text],loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.5,1.5)
        
        return self.fig

    def show_test_metrics(self,train_record, test_record):
      
        self.fig.clf()
        self.fig.set_size_inches(2,1.5)
        cols = train_record[0]
        train_row = train_record[1]
        test_row = test_record[1]
        
        rows = ["Training","Testing"]
        axs = self.fig.subplots()
        self.fig.suptitle("Model Performance")
        self.fig.canvas.header_visible = False
        self.fig.patch.set_visible(False)
        axs.axis('off')
        axs.axis('tight')
        self.fig.tight_layout()
        table = axs.table( colLabels=cols,rowLabels=rows,cellText=[train_row, test_row],loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.5,1.5)
        
        return self.fig

class ViewROC:

    def __init__(self):
        self.fig = plt.figure(figsize=(4,3),layout="constrained")


    def show_ROC(self,train_model,features_train,features_test,labels_train,labels_test,test_model,test_mode):

        self.fig.clf()
        self.fig.set_size_inches(4,3)
        ax = self.fig.subplots()
        ax.set_title("ROC Curve")
        RocCurveDisplay.from_estimator(train_model, features_train, labels_train,ax=ax,name="train") 
        
        if test_mode == 1:
            pred = test_model.decision_function(features_test)
            RocCurveDisplay.from_predictions(labels_test,pred,ax=ax,name="test")
        
        self.fig.canvas.header_visible = False
        ax.set_ylabel('True Positive Rate')
        ax.set_xlabel('False Positive Rate')
        self.fig.tight_layout()
        return self.fig





class ViewDET:
    
    def __init__(self):
        self.fig = plt.figure(figsize=(3,3), layout="constrained")

    def show_DET(self,train_model,features_train,features_test,labels_train,labels_test,test_model,test_mode):
        self.fig.clf()
        self.fig.set_size_inches(3,3)
        ax = self.fig.subplots()
        ax.set_title("DET Curve")
        self.fig.canvas.header_visible = False
        DetCurveDisplay.from_estimator(train_model, features_train, labels_train,ax=ax,name='train')
        if test_mode == 1:
            pred = test_model.decision_function(features_test)
            DetCurveDisplay.from_predictions(labels_test,pred,ax=ax,name='test')
      
        ax.set_ylabel('False Negative Rate')
        ax.set_xlabel('False Positive Rate')
        self.fig.tight_layout()
        return self.fig
    
