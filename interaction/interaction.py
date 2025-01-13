from utils.data_utils import tsne
from processData.AttributeSelection import AttributeSelection
from interaction.ui_utils import get_correctness, data_filer, get_data_attr_distribution,show_image
import numpy as np
from ipywidgets import Output, HBox, widgets, VBox, Layout, Tab
from plotly import graph_objects as go
from interaction.activeLearning import LeastConfidence, Entropy, Margin, Coreset, KMeansSampling
from utils.defines import *
from processData.Corpus import emptyAttribute
import json

display = Output()

class Interaction():
    def __init__(self, datas, labels, predictions, embeddings, tags=None, shuffle=True, label_names_dict={}, attributes_dict_path='', error_list=[]) -> None:
        """
        Classification Task Only

        datas (n * [image shape]): list of images / image address
        labels (n): the labels (int) for each img, NO_LABEL for no label data
        predictions (n * num of classes)
        embeddings (n * feature size): features of data
        tags: list of tag, TRAIN, VALID, UNLABELED
        shuffle: shuffle data display order
        attributes_dict: attributes from pre-processing
        error_list: error types for each failure
        """
        assert len(datas)==len(labels)==len(predictions)==len(embeddings),  'Inputs should have equal length'
        assert len(datas)!=0,  'Need more than one data'
        if tags == None: tags = [TRAIN if label!=NO_LABEL else UNLABELED for label in labels]
        self.datas = datas
        self.display_index = np.arange(len(self.datas))
        self.labels = [str(i) for i in labels]
        self.predictions = predictions
        self.predicted_labels = [str(i) for i in np.argmax(predictions, axis=1)] if len(predictions.shape) > 1 else predictions
        self.label_names_dict = label_names_dict if label_names_dict!={} else {label:label for label in set(labels)}
        self.correctness = get_correctness(self.predicted_labels,self.labels)
        self.tags = tags
        self.attributes_dict = json.load(open(attributes_dict_path))
        self.error_list = error_list
        if embeddings.shape[1] != 2: embeddings = tsne(embeddings)
        self.embedding = embeddings
        if shuffle: np.random.shuffle(self.display_index)
        self.selected_node_index = self.display_index[0]
        self.selected_node_index_list = {}
        self.selected_data_index_list = []
        self.recent_display_index = {}
        self.filter = data_filer(self.predicted_labels,self.labels,self.tags,self.attributes_dict)
        self.unlabeled_length = len(self.filter.filter(UNLABELED))
        self.captions_list = ['' for i in range(len(datas))]

    def interaction(self):
        """
        Graph Control | Data Control
        Scatter/Bar   | Analysis/Selection/Display
        """
        self.create_graph()
        self.create_data()
        self.update_attributes_layout()
        self.data_tab.layout = Layout(max_width='40%')
        return HBox([self.graph, self.data_tab])

    def create_graph(self):
        dataset_options = [ALL,TRAIN,VALID,UNLABELED]
        self.graph_dataset_choice = widgets.Dropdown(
            options=dataset_options,
            value=ALL,
            description='Display:',
        )
        data_options = [ALL, 'Failure', 'Uncertain Data','Selected','Unselected']
        self.graph_data_choice = widgets.Dropdown(
            options=data_options,
            value=ALL,
            description='',
        )
        attributes_options = ['Dataset','Prediction','Correctness',] + list(self.attributes_dict.keys())
        self.graph_attributes_choice = widgets.Dropdown(
            options=attributes_options,
            value='Dataset',
            description='Color By:',
        )
        self.graph_tab = Tab()
        self.graph_scatter = go.FigureWidget()
        self.graph_scatter_slider = widgets.FloatSlider(value=1, min=0, max=100, step=0.1, description='Show%:', orientation='horizontal')
        self.graph_bar = go.FigureWidget()
        bar_options = ['Prediction','Correctness',] + list(self.attributes_dict.keys())
        self.graph_distribution_choice = widgets.Dropdown(
            options=bar_options,
            value='Prediction',
            description='Distribution:',
        )
        children = [VBox([self.graph_scatter_slider,self.graph_scatter]),VBox([self.graph_distribution_choice,self.graph_bar])]
        self.graph_tab.children = children
        self.graph_tab.set_title(0, 'Scatter')
        self.graph_tab.set_title(1, 'Histogram')
        self.graph_set_button = widgets.Button(description='Set',button_style='success')
        self.graph_set_button.on_click(self.on_graph_set_button)
        self.graph = VBox([HBox([self.graph_dataset_choice,self.graph_data_choice]),HBox([self.graph_attributes_choice,self.graph_set_button]),self.graph_tab])
    
    def create_data(self):
        self.create_dataAnalysis()
        self.create_dataSelection()
        self.create_dataDisplay()
        tab1 = VBox([HBox([self.attr_add_button,self.attr_delete_button,self.attr_apply_button]),self.attr_new_text,self.attr_accordion])
        tab2 = VBox([HBox([self.selection_add_button,self.selection_delete_button,self.selection_reset_button]),self.selection_text,HBox([self.selection_budget_slider,self.selection_save_button]),VBox([self.selection_attr_checkbox, self.selection_AL_checkbox]), self.selection_attr_accordion, self.selection_AL_choice])
        data_contents = ['Data Analysis', 'Data Selection','Display Data']
        self.data_tab = Tab()
        self.data_tab.children = [tab1,tab2,self.display_out]
        for i in range(len(self.data_tab.children)):
            self.data_tab.set_title(i, data_contents[i])

    def create_dataAnalysis(self):
        self.attr_add_button = widgets.Button(description='Add',button_style='success')
        self.attr_delete_button = widgets.Button(description='Remove',button_style='warning')
        self.attr_apply_button = widgets.Button(description='Auto Label',button_style='success')
        self.attr_accordion = widgets.Accordion(children=[])
        self.attr_new_text = widgets.Text(description='Attribute:',placeholder='Try to separate correct/failure')
        self.attr_add_button.on_click(self.on_attr_add_button)
        self.attr_delete_button.on_click(self.on_attr_delete_button)
        self.attr_apply_button.on_click(self.on_attr_apply_button)

    def create_dataSelection(self):
        self.selection_budget_slider = widgets.FloatSlider(value=0, min=0, max=100, step=0.1, description='Budget%:', orientation='horizontal')
        self.selection_add_button = widgets.Button(description='Add',button_style='success')
        self.selection_delete_button = widgets.Button(description='Remove',button_style='warning')
        self.selection_reset_button = widgets.Button(description='Reset',button_style='success')
        self.selection_save_button = widgets.Button(description='Export',button_style='warning')
        self.selection_text = widgets.Label('Total number of selected data: 0 (0%)')

        self.selection_attr_accordion = widgets.Accordion(children=[])
        widget_list = []
        distinct_attrs = [self.attributes_dict[key]['word'] for key in self.attributes_dict.keys()]
        for descriptions in distinct_attrs:
            sub_attr_options = [ALL,] + [text for text in descriptions]
            sub_attr_choice = widgets.SelectMultiple(
                options=sub_attr_options,
                description='',
            )
            widget_list.append(sub_attr_choice)
        self.selection_attr_accordion.children = widget_list
        for i in range(len(self.selection_attr_accordion.children)):
            self.selection_attr_accordion.set_title(i, list(self.attributes_dict.keys())[i])

        AL_options = [('Least Confidence', 0),('Entropy',1),('Margin Score',2),('Coreset',3),('KMeans',4)]
        self.selection_AL_choice = widgets.Dropdown(
            options=AL_options,
            value=0,
            description='AL method:',
        )

        self.selection_AL_methods = [LeastConfidence, Entropy, Margin, Coreset, KMeansSampling]
        self.selection_attr_checkbox = widgets.Checkbox(description='Filter By Attributes')
        self.selection_AL_checkbox = widgets.Checkbox(description='Apply Active Learning')
        self.selection_add_button.on_click(self.on_selection_add_button)
        self.selection_delete_button.on_click(self.on_selection_delete_button)
        self.selection_reset_button.on_click(self.on_selection_reset_button)
        self.selection_save_button.on_click(self.on_selection_save_button)

    def create_dataDisplay(self):
        self.display_out = display

    def get_selection_data(self):
        nodes = []
        unlabeled_index = self.filter.filter(selected_dataset=UNLABELED)
        selected_number = int(self.selection_budget_slider.value*len(unlabeled_index)/100)
        if self.selection_attr_checkbox.value:
            selected_attrs = []
            for child in self.selection_attr_accordion.children:
                selected_attrs.append(child.value)
            nodes = self.filter.filter(UNLABELED,'Unselected',selected_attrs,self.selected_data_index_list)
            unlabeled_index = nodes
        else:
            nodes = unlabeled_index
        if self.selection_AL_checkbox.value:
            nodes_for_AL = self.filter.filter(TRAIN) + self.filter.filter(VALID) + unlabeled_index
            nodes_for_AL = np.array(nodes_for_AL)
            predictions = self.predictions[nodes_for_AL]
            embedding = self.embedding[nodes_for_AL]
            tags = [self.tags[i] for i in nodes_for_AL]
            index = self.selection_AL_methods[self.selection_AL_choice.value](predictions,embedding,tags,selected_number)
            nodes = nodes_for_AL[index]
        nodes = list(nodes)
        selected_on_graph = []
        for graph_selected in self.selected_node_index_list.values():
            selected_on_graph += graph_selected
        nodes = nodes[:selected_number] + selected_on_graph
        return nodes

    def on_graph_scatter_click(self, trace, points, state):
        if len(points.point_inds):
            ind = points.point_inds[0]
            self.selected_node_index = self.recent_display_index[trace.name][ind]
            self.display_image()
            self.update_dataAnalysis_attr_accordian()

    @display.capture()
    def on_graph_scatter_select(self,trace, points, state):
        self.selected_node_index_list[trace.name] = [self.recent_display_index[trace.name][i] for i in points.point_inds]

    def on_graph_set_button(self,sender):
        self.update_graph_bar()
        self.update_graph_scatter()

    def on_selection_add_button(self, sender):
        nodes = self.get_selection_data()
        self.selected_data_index_list = list(set(self.selected_data_index_list+nodes))
        self.update_selection_text()

    def on_selection_delete_button(self, sender):
        nodes = self.get_selection_data()
        self.selected_data_index_list = [i for i in self.selected_data_index_list if i not in nodes]
        self.update_selection_text()

    def on_selection_reset_button(self, sender):
        self.selected_data_index_list = []
        self.update_selection_text()

    def on_selection_save_button(self, sender):
        np.save('',self.selected_data_index_list)

    def on_attr_add_button(self, sender):
        self.attributes_dict[self.attr_new_text.value] = emptyAttribute()
        self.attributes_dict[self.attr_new_text.value]['data'] = ['' for i in range(len(self.datas))]
        self.update_attributes_layout()
    
    def on_attr_delete_button(self, sender):
        selected_idx = self.attr_accordion.selected_index
        name = self.attr_accordion.get_title(selected_idx)
        del self.attributes_dict[name]
        self.update_attributes_layout()

    def on_attr_apply_button(self,sender):
        AP = AttributeSelection(self.datas,self.labels,self.attributes_dict, self.label_names_dict)
        AP.match_description_to_data()
        self.attributes_dict = AP.attributes_dict
        self.update_attributes_layout()

    def on_attr_save_text(self,sender):
        selected_idx = self.attr_accordion.selected_index
        value = self.attr_accordion.children[selected_idx].value
        name = self.attr_accordion.get_title(selected_idx)
        self.attributes_dict[name]['data'][self.selected_node_index] = value
        self.attributes_dict[name]['word'] = set(self.attributes_dict[name]['data'])
        self.update_graph_scatter()
        self.update_graph_bar()
        self.update_selection_attr_accordion()

    def update_attributes_layout(self):
        self.update_dataAnalysis_attr_accordian()
        self.update_selection_attr_accordion()
        self.update_graph_attributes_choice()
        self.update_graph_scatter()
        self.update_graph_bar()

    def update_dataAnalysis_attr_accordian(self):
        data_idx = self.selected_node_index
        widget_list = []
        attributes_list = [self.attributes_dict[attribute]['data'] for attribute in self.attributes_dict.keys()]
        attributes_list = [[attributes_list[i][j] for i in range(len(attributes_list))] for j in range(len(self.datas))]
        for i, text in enumerate(attributes_list[data_idx]):
            widget = widgets.Text(text,placeholder='Press Enter to Save')
            widget.on_submit(self.on_attr_save_text)
            widget_list.append(widget)
        self.attr_accordion.children = widget_list
        for i in range(len(self.attr_accordion.children)):
            self.attr_accordion.set_title(i, list(self.attributes_dict.keys())[i])

    def update_selection_text(self):
        self.selection_text.value = 'Total number of selected data: %d (%.2f%%)'%(len(self.selected_data_index_list),100*len(self.selected_data_index_list)/self.unlabeled_length)

    def update_selection_attr_accordion(self):
        widget_list = []
        distinct_attrs = [self.attributes_dict[key]['word'] for key in self.attributes_dict.keys()]
        for descriptions in distinct_attrs:
            sub_attr_options = [ALL,] + [text for text in descriptions if text!='']
            sub_attr_choice = widgets.SelectMultiple(
                options=sub_attr_options,
                description='',
            )
            widget_list.append(sub_attr_choice)
        self.selection_attr_accordion.children = widget_list
        for i in range(len(self.selection_attr_accordion.children)):
            self.selection_attr_accordion.set_title(i, list(self.attributes_dict.keys())[i])

    def update_graph_attributes_choice(self):
        self.graph_attributes_choice.options = ['Dataset','Prediction','Correctness',] + list(self.attributes_dict.keys())

    def update_graph_scatter(self):
        selected_dataset = self.graph_dataset_choice.value
        display_data = self.graph_data_choice.value
        attribute = self.graph_attributes_choice.value
        display_percentage = self.graph_scatter_slider.value
        index = self.filter.filter(selected_dataset, display_data, selected_data_index=self.selected_data_index_list, indexs=self.display_index, recent_attribute=attribute)
        display_number = int(len(index) * display_percentage / 100)
        index = index[:display_number]
        attributes = self.filter.get_attribute(attribute,index)
        scatters_name = list(set(list(attributes)))
        self.recent_display_index = {}
        self.selected_node_index_list = {}
        attributes_list = [self.attributes_dict[attribute]['data'] for attribute in self.attributes_dict.keys()]
        attributes_list = [[attributes_list[i][j] for i in range(len(attributes_list))] for j in range(len(self.datas))]
        for name in scatters_name:
            scatter_data = [index[i] for i in range(len(index)) if attributes[i]==name]
            self.recent_display_index[name]=scatter_data
            predictions, labels, attrs = [self.predicted_labels[i] for i in scatter_data],[self.labels[i] for i in scatter_data],[attributes_list[i] for i in scatter_data]
            label_text = ['Prediction: ' + str(i)  + '\nLabel: ' + str(j) + '\nAttr: ' + str(k) for i,j,k in zip(predictions, labels, attrs)]
            X, Y = self.embedding[scatter_data][:,0],self.embedding[scatter_data][:,1]
            scatter = go.Scatter(x=X, y=Y, mode='markers',hovertext=label_text, name=name)
            self.graph_scatter.add_trace(scatter)

        self.graph_scatter.data = self.graph_scatter.data[-len(scatters_name):]
        for i in range(len(scatters_name)):
            self.graph_scatter.data[i].on_click(self.on_graph_scatter_click)
            self.graph_scatter.data[i].on_selection(self.on_graph_scatter_select)

    def update_graph_bar(self):
        selected_dataset = self.graph_dataset_choice.value
        display_data = self.graph_data_choice.value
        distribution = self.graph_distribution_choice.value
        index = self.filter.filter(selected_dataset,display_data, selected_data_index=self.selected_data_index_list)
        bar = self.graph_attributes_choice.value
        if bar == 'Prediction':
            attributes = self.predicted_labels
            bars = list(set(list(self.predicted_labels)))
        elif bar == 'Dataset':
            attributes = self.tags
            bars = [TRAIN,VALID,UNLABELED]
        elif bar == 'Correctness':
            attributes = self.correctness
            bars = [CORRECT,WRONG,UNKNOWN]
        else:
            attributes = self.filter.get_attribute(bar)
            bars = list(set(attributes))

        for target in bars:
            target_index = [i for i in index if attributes[i]==target]
            attributes_for_distribution = self.filter.get_attribute(distribution, target_index)
            distinct_attrs, count_attrs = get_data_attr_distribution(attributes_for_distribution)
            bar = go.Bar(x=distinct_attrs, y=count_attrs, name=target)
            self.graph_bar.add_trace(bar)

        self.graph_bar.data = self.graph_bar.data[-len(bars):]

    @display.capture(clear_output=True)
    def display_image(self):
        img = self.datas[self.selected_node_index]
        show_image(img)
        print("Node :", self.selected_node_index, "Prediction:", self.predicted_labels[self.selected_node_index], "\tLabel:", self.labels[self.selected_node_index])
        for name in self.attributes_dict.keys():
            print('\n%s: %s'%(name, self.attributes_dict[name]['data'][self.selected_node_index]))
        if len(self.error_list):
            for error in self.error_list[self.selected_node_index]:
                print('Potential Error: ',error)

    def display_static(self):
        pass