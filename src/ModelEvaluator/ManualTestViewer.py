import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display
from functools import partial
import matplotlib.pyplot as plt
import src.FeatureExplorer.SampleDataController as datacontrol
import pandas as pd

import src.FeatureExplorer.FeatureExtractors as fe
import numpy as np


def get_single_prediction(df,model):
    
    record = df.get_current_record()

    if record[0] == -1:
        return 'No more records in manual test set!'
    
    scaler = df.scaler
    af = fe.AudioFeatures(record)
    mfcc = af.get_mfcc_as_npy(df.mfcc_filter_count)
    mel = af.get_mel_as_numpy(df.mel_filter_count)
    features = np.hstack((mfcc, mel))
    features = scaler.transform(features[np.newaxis,:])
    
    if model.predict(features) == 1:
        return 'angry'
    else:
        return 'not angry'



def get_single_matrix(df,record):
    scaler = df.scaler
    af = fe.AudioFeatures(record)
    mfcc = af.get_mfcc_as_npy(40)
    mel = af.get_mel_as_numpy(128)

    features = np.hstack((mfcc, mel))
    features = scaler.transform(features[np.newaxis,:])

    return features


def view_manual_controls(df,tr):

# Layouts

    left_align = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='flex-start',
                    border='none',
                    width='auto')

    center_align = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='center',
                    width='100%')

    justify = widgets.Layout(display='flex',
                    flex_flow='row',
                    justify_content='space-around',
                    width='75%',
                    margin='0px 15px 0px 55px')

    md_layout = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_content='flex-start',
                    align_items='flex-start',
                    #margin='0px 15px 0px 55px',
                    justify_content='flex-start')

    ds_layout = widgets.Layout(display='flex',
                    flex_flow='row',
                    align_content='center',
                    align_items='center',
                    width='100%',
                    justify_content='center',
                    margin='15px 0px 15px 0px')


    # Filters 
    # Actor Emotion
    emotion_menu_label = widgets.Label(value="Emotions",
                                       style=dict(font_weight='bold'))
    
    emotion_set =['all','neutral', 'calm', 'happy','sad','angry','fearful','disgust','surpised']
    emotion_menu = widgets.SelectMultiple(options=emotion_set,
                                            value=['all'],
                                            description='',
                                            disabled=False)

    # Actor Sex
    actor_sex_menu_label =widgets.Label(value="Actor Sex",
                                        style=dict(font_weight='bold'))
    
    actor_sex_menu = widgets.Dropdown(options=['all','male', 'female'],
                                        value='all',
                                        description='',
                                        disabled=False,
                                        layout = widgets.Layout(grid_area='actor_sex'))

    # Actor ID
    actor_id_menu_label = widgets.Label(value="Actor ID",
                                        style=dict(font_weight='bold'))
    actors_id_list = ['all']
    for i in range(2):
        actors_id_list.append(str(i+1))
    actor_ids_menu = widgets.SelectMultiple(options=actors_id_list,
                                            value=['all'],
                                            description='',
                                            disabled=False)
    # Feature Extraction Settings
    num_mel_filters = widgets.FloatLogSlider(value=128,
                                            base=2,
                                            min=3, # min exponent of base
                                            max=8, # max exponent of base
                                            step=1, # exponent step
                                            description='# mel filters')
    
    num_mfcc_filters = widgets.IntSlider(value=32,
                                        min=20, # min exponent of base
                                        max=120, # max exponent of base
                                        step=20, # exponent step
                                        description='# mfcc filters')

    # Send filter settings to controller and request new data
    update_filters = widgets.Button(description='Apply Filters',
                                    layout=Layout(align='right'),
                                    style=dict(
                                    button_color='blue',
                                    text_color='white',
                                    font_weight='bold'))

    # Get the next record in the filtered subset
    nextbutton = widgets.Button(description='Next Record',
                                style=dict(
                                button_color='darkgreen',
                                text_color='white',
                                font_weight='bold'))

    # show the sample number and the total count according to the filter
    sample_position_in_set = widgets.Label(value="Sample : "+ 
                                        str(df.get_current_record_number()) +
                                        " of "+ 
                                        str(118),
                                        layout=left_align)



    prediction = widgets.Label(value='not angry',
                               layout=ds_layout,
                               style=dict(text_color='black',
                                        font_weight='bold'))



    # Outputs
    sample_metadata_record = widgets.Output()
    audio_player_output = widgets.Output()
    prediction_out = widgets.Output()
    spectrograms_output = widgets.Output() 

    # Initialize
    df.apply_filters(emotion_list=emotion_menu.value, sex=actor_sex_menu.value,ids=actor_ids_menu.value)
    



    # initial record display on load
    def initial_outputs(dfx):
        with sample_metadata_record:
            display(dfx.show_record_metadata(),sample_position_in_set)

        with audio_player_output:    
            display(dfx.get_record_audio())

        with prediction_out:
            prediction.value = ("Prediction: "  + get_single_prediction(dfx,tr.get_model_used_for_test()))
            display(prediction)

        with spectrograms_output:
            fig = dfx.get_record_viz(df.mel_filter_count, 8000,df.mel_filter_count) 
            display(fig.canvas)
        

    initial_outputs(df)



    # Handlers
    def update_filters_handler(dfx, w):
       
        with sample_metadata_record:     
            sample_metadata_record.clear_output(wait=True)
            dfx.apply_filters(emotion_list=emotion_menu.value, sex=actor_sex_menu.value,ids=actor_ids_menu.value)
            sample_position_in_set.value="Sample : "+ str(dfx.get_current_record_number()) +" of "+ str(dfx.get_num_samples())  
            display(dfx.show_record_metadata(),sample_position_in_set)
        
        with prediction_out:
            prediction_out.clear_output(wait=True)
            prediction.value = ("Prediction: "  + get_single_prediction(dfx,tr.get_model_used_for_test()))
            display(prediction)

        with audio_player_output:
            audio_player_output.clear_output(wait=True)    
            display(df.get_record_audio())

        with spectrograms_output:
            spectrograms_output.clear_output(wait=True) 
            fig = dfx.get_record_viz(df.mel_filter_count, 8000,df.mel_filter_count)   
            display(fig.canvas)
        
    def next_record_handler(dfx, next_button):
        
        with sample_metadata_record:
            dfx.get_next_record()
            sample_metadata_record.clear_output(wait=True)
            sample_position_in_set.value="Sample : "+ str(dfx.get_current_record_number()) +" of "+ str(dfx.get_num_samples())  
            
            display(dfx.show_record_metadata(),sample_position_in_set)
        
        with prediction_out:
            prediction_out.clear_output(wait=True)
            prediction.value = ("Prediction: "  + get_single_prediction(dfx,tr.get_model_used_for_test()))
            display(prediction)
        
        with audio_player_output:
            audio_player_output.clear_output(wait=True)    
            display(dfx.get_record_audio())

        with spectrograms_output:
            spectrograms_output.clear_output(wait=True)
            fig = dfx.get_record_viz(df.mel_filter_count, 8000,df.mel_filter_count)   
            display(fig.canvas)



    # listeners
    update_filters.on_click(partial(update_filters_handler, df))
    nextbutton.on_click(partial(next_record_handler, df))




    # Containers

    audio_player_box = widgets.HBox(children=[audio_player_output])
    metadata_box = widgets.VBox(children=[sample_metadata_record])
    buttons_box = widgets.HBox(children=[update_filters, nextbutton])

    pred_box = widgets.VBox([prediction_out])

    control_box = widgets.VBox([emotion_menu_label,
                            emotion_menu,
                            actor_sex_menu_label,
                            actor_sex_menu,
                            actor_id_menu_label,
                            actor_ids_menu,
                            buttons_box,
                            metadata_box,
                            audio_player_box,
                            pred_box
                            ],layout=md_layout)

    view_box = widgets.Box(children=[spectrograms_output])
    return widgets.HBox([control_box,view_box],layout=justify)