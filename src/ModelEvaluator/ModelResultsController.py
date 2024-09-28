from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay,DetCurveDisplay
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score,precision_score,accuracy_score,d2_absolute_error_score
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import src.util.RecordVisualizer as rv



from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score,precision_score,accuracy_score,d2_absolute_error_score
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt


class ResultsViewer:

    def __init__(self, df, sample_data):
        
        self.df = self.arrange_columns(df)
        
        self.model = SVC()
        self.current_record_idx = 0

  

        self.features_train = sample_data.features_train
        self.labels_train = sample_data.labels_train

        # Visualizers
        self.vgr = rv.ViewGridSearchResults()
        self.prc = rv.ViewPrecisionRecall()
        self.vcm = rv.ViewConfusionMatrix()
        self.vmm = rv.ViewModelMetrics()
        self.vroc = rv.ViewROC()
        self.vdet = rv.ViewDET()




        # Testing only
        self.features_test = sample_data.features_test
        self.labels_test = sample_data.labels_test
        self.test_model = None
        self.test_mode = 0

        self.predictions = []


    def arrange_columns(self, in_df):
        df = in_df.copy()
        df['rank']=list(range(1,len(df)+1))
        df.drop(columns=['rank_test_score'],inplace=True)
        df.columns=['L2Mult','weight','gamma','test_recall','train_recall', 'test_stdev','rank']
        df = df[['rank','L2Mult','weight','gamma','test_recall','train_recall', 'test_stdev']]
        return df
    

    def get_top_records(self, num_records=10):
        return self.df.iloc[0:num_records]

    def format_top_records_table(self, recs,num_records=10):

        cols = ['rank', 'weight','gamma','L2Mult','test recall', 'train recall']

        cell_text = []
        for i in range(num_records):
            
            weight = 'bal'
            if recs['weight'].iloc[i] == None:
                weight ="unbal"
            
            cell_text.append([recs['rank'].iloc[i], 
                              weight,
                              recs['gamma'].iloc[i],
                              round(recs['L2Mult'].iloc[i],7),
                              round(recs['test_recall'].iloc[i],2),
                              round(recs['train_recall'].iloc[i],2)])
        return cols, cell_text

    def print_records_to_table(self):
        return self.vgr.print_records_to_table(self.format_top_records_table(self.get_top_records()))

  ####################################################################################
    
    def select_record(self,idx):
        plt.close()
        self.current_record_idx = idx
        self.fit_model_with_record(idx)

    
    def apply_record_to_model(self, rec_index=0):
        
        return SVC(C=self.df['L2Mult'].iloc[rec_index],
                   kernel='rbf',
                   gamma=self.df['gamma'].iloc[rec_index],
                   class_weight=self.df['weight'].iloc[rec_index]
                   )
     
           
    def fit_model_with_record(self, rec_index=0):    
        self.model = self.apply_record_to_model(rec_index=rec_index)
        self.model = self.model.fit(self.features_train,self.labels_train)
        return self.model
       

    def set_testing_model(self):
        self.test_model = self.model
        self.test_mode = 1
        self.predictions = self.test_model.predict(self.features_test)

    def get_model_used_for_test(self):
        return self.test_model


    def show_confusion_matrix_train(self):
        return self.vcm.show_confusion_matrix_train(self.model,
                                                    self.features_train,
                                                    self.labels_train)
   

    def show_confusion_matrix_test(self):
        return self.vcm.show_confusion_matrix_test(self.test_mode,
                                                   self.labels_test,
                                                   self.predictions)
    
    def show_ROC(self):
        return self.vroc.show_ROC(self.model,
                                self.features_train,
                                self.features_test,
                                self.labels_train,
                                self.labels_test,
                                self.test_model,
                                self.test_mode)
        
      
    def show_DET(self):
        return self.vdet.show_DET(self.model,
                                self.features_train,
                                self.features_test,
                                self.labels_train,
                                self.labels_test,
                                self.test_model,
                                self.test_mode)


    def show_precision_recall(self):
        return self.prc.show_precision_recall(self.model,
                                              self.features_train,
                                              self.features_test,
                                              self.labels_train,
                                              self.labels_test,
                                              self.test_model,
                                              self.test_mode)


    def get_train_metrics(self):
      
        scoring = ['recall','precision','accuracy','d2_absolute_error_score']
        scores = cross_validate(self.model, self.features_train, self.labels_train,scoring=scoring)
        
        cols = ['recall','precision','accuracy','d2_error']
        cells = []
        cells.append(round(scores['test_recall'].mean(),2))
        cells.append(round(scores['test_precision'].mean(),2))
        cells.append(round(scores['test_accuracy'].mean(),2))
        cells.append(round(scores['test_d2_absolute_error_score'].mean(),2))

        return cols, cells


    def get_test_metrics(self):
        pred = self.test_model.decision_function(self.features_test)
        cols = ['recall','recall_micro','precision','d2_error']
        cells = []
        cells.append(round(recall_score(self.labels_test, self.predictions),3))
        cells.append(round(precision_score(self.labels_test, self.predictions),3))
        cells.append(round(accuracy_score(self.labels_test, self.predictions),3))
        cells.append(round(d2_absolute_error_score(self.labels_test,self.predictions),3))
        
        return cols, cells
    
    def show_train_metrics(self):
        return self.vmm.show_train_metrics(self.get_train_metrics())


    def show_test_metrics(self):
        return self.vmm.show_test_metrics(self.get_train_metrics(),self.get_test_metrics())