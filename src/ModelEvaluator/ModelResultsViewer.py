import ipywidgets as widgets
from ipywidgets import Layout, AppLayout
from IPython.display import display
from functools import partial



#@title Viewer
def eval_controller(tr):


    # layouts

    stretch_records = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='stretch',
                    width='30%')

    stretch_roc = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='flex-start',
                    align_content='center',
                    justify_content='flex-start',
                    width='30%',
                    margin='0px 50px 0px 0px')

    stretch_stats = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='stretch',
                    width='30%',
                    margin='10px 0px 0px 0px')

    center_align = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='center',
                    width='100%')


    stat_box_layout = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='stretch',
                    align_content='center',
                    width='90%',
                    justify_content='center',
                    margin='0px 0px 0px 0px')

    top_row_layout = widgets.Layout(display='flex',
                    flex_flow='row',
                    align_content='flex-start',
                    width='100%',
                    justify_content='flex-start')
    
    bottom_row_layout = widgets.Layout(display='flex',
                    flex_flow='row',
                    align_content='flex-start',
                    width='100%',
                    justify_content='flex-start')

    prec_test_layout = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_content='flex-start',
                    #align_items='stretch',
                    width='30%',
                    #justify_content='flex-start',
                    margin='10px 50px 0px 0px')

    cm_layout = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_content='stretch',
                    align_items='stretch',
                    width='30%',
                    justify_content='flex-start',
                    margin='0px 0px 0px 20px')
    cm2_layout = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_content='stretch',
                    align_items='stretch',
                    width='30%',
                    justify_content='flex-start',
                    margin='10px 0px 0px 20px')

    det_layout = widgets.Layout(display='flex',
                   
                    flex_flow='row',
                    align_content='center',
                    align_items='center',
                    width='30%',
                    justify_content='center',
                    margin='10px 0px 0px 0px')

    oos_test_layout = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='flex-start',
                    width='80%',
                    justify_content='flex-start',
                    margin='60px 0px 0px 0px')




    record_out = widgets.Output(layout=stretch_records)
    roc_out = widgets.Output(layout=stretch_roc)
    stats_out =  widgets.Output()
    det_out = widgets.Output(layout=det_layout)

    cm_out = widgets.Output(layout=cm_layout)
    cm2_out = widgets.Output(layout=cm2_layout)
    prec_recall_out = widgets.Output()
    test_metrics_out = widgets.Output()





    # widgets
    rec_sel_dropdown_layout = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='flex-start',
                    align_content='flex-start',
                    width='90%',
                    margin='0px 0px 20px 0px')

    rec_sel_label = widgets.Label(value="Select record by rank to view performance metrics",style=dict(font_weight='bold'))

    rec_sel_dropdown = widgets.Dropdown(
        options=['1','2','3','4','5','6','7','8','9','10'],
        value='1',
        description='',
        disabled=False
    )

    run_oos_button  = widgets.Button(description='Step 5: Test model on out of sample data',layout=oos_test_layout,style=dict(
    button_color='darkred',
    text_color='white',
    font_weight='bold'
    ))



    # Containers

    record_selection_box = widgets.HBox([rec_sel_label,rec_sel_dropdown],layout=rec_sel_dropdown_layout)

    row_one_chart_box = widgets.HBox([record_out, roc_out,cm2_out],layout=top_row_layout)

    #cm_box = widgets.HBox([cm_out,cm2_out],layout=cm_layout)

    ll_box = widgets.VBox([stats_out,run_oos_button],layout=stretch_stats)

    prec_test_box = widgets.VBox([prec_recall_out],layout=prec_test_layout)

    row_two_chart_box = widgets.HBox([ll_box,prec_test_box,cm_out])



    def initialize():
        tr.fit_model_with_record(0)

        with record_out:

            record_out.clear_output(wait=True)
            display(tr.print_records_to_table())

        with stats_out:
            stats_out.clear_output(wait=True)
            display(tr.show_train_metrics())

        with cm2_out:
            cm2_out.clear_output(wait=True)

        with cm_out:
            cm_out.clear_output(wait=True)
            #display(tr.show_confusion_matrix_train())

        with roc_out:
            roc_out.clear_output(wait=True)
            display(tr.show_ROC())

        #with det_out:
         #   det_out.clear_output(wait=True)
          #  display(tr.show_DET())

        with prec_recall_out:
            prec_recall_out.clear_output(wait=True)
            display(tr.show_precision_recall())



    def select_record_to_view(dfx,names):
      
        val = int(names.new) - 1
        dfx.select_record(val)

        with stats_out:
            stats_out.clear_output(wait=True)
            display(dfx.show_train_metrics())

        with cm_out:
            cm_out.clear_output(wait=True)
            #display(dfx.show_confusion_matrix_train())

        with roc_out:
            roc_out.clear_output(wait=True)
            display(dfx.show_ROC())

        #with det_out:
         #   det_out.clear_output(wait=True)
          #  display(dfx.show_DET())

        with prec_recall_out:
            prec_recall_out.clear_output(wait=True)
            display(dfx.show_precision_recall())




    def test_model_on_oos(dfx,val):

        dfx.set_testing_model()

        run_oos_button.description="Showing model performance on test set"
        run_oos_button.disabled=True
        run_oos_button.style.button_color="black"

        with stats_out:
            stats_out.clear_output(wait=True)
            display(dfx.show_test_metrics())

        with cm_out:
            cm_out.clear_output(wait=True)
            #display(dfx.show_confusion_matrix_train())
        
        with cm2_out:
            cm2_out.clear_output(wait=True)
            display(dfx.show_confusion_matrix_test())

        with roc_out:
            roc_out.clear_output(wait=True)
            display(dfx.show_ROC())

        #with det_out:
         #   det_out.clear_output(wait=True)
          #  display(dfx.show_DET())

        with prec_recall_out:
            prec_recall_out.clear_output(wait=True)
            display(dfx.show_precision_recall())

        with test_metrics_out:
            test_metrics_out.clear_output(wait=True)








    initialize()



    rec_sel_dropdown.observe(partial(select_record_to_view,tr),names='value')
    run_oos_button.on_click(partial(test_model_on_oos, tr))



    return widgets.VBox([record_selection_box,row_one_chart_box,row_two_chart_box],layout=stat_box_layout)
