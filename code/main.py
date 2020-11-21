# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:08:38 2019

@author: asadc
"""
'''
Main class calls the GA algorithm for each dataset
with different network models and prints performance measures.
'''

import LoadDataset as ld
import GeneticAlg
import Metrics
import pandas as pd
import numpy as np
from graphs import plot_graph

class main:
        
        def __init__(self):
                #define variable to use inside class which may need tuning
                self.splitlength = 0.75  
                self.layers = [0, 1, 2]
                self.alldataset = ld.LoadDataset().load_data()          #load all dataset
                #check if dataset is classification
                self.IsClassificationDict = ld.LoadDataset().IsClassificationDict() 
                #define dataframe to store all the results
                self.allresults = pd.DataFrame(columns=['dataset', 'isClassification', 'hiddenLayers', 'method',
                                                        'accuracy', 'precision', 'recall', 'NRMSE'])
                self.algorithms = ["GA"]            
        
        def main(self):
                # for each dataset call each algorithm and number of hidden layers
                for dataset in self.alldataset:         
                        print('current dataset ::: {0} \n'.format(dataset))
                        data = self.alldataset.get(dataset)
                        isClassification = self.IsClassificationDict.get(dataset)
                        trainset, testset = self.testtrainsplit(data, self.splitlength)
                        len_input_neuron = len(data.columns[:-1])
                        len_output_neuron = 1
                        if isClassification: len_output_neuron = len(data[data.columns[-1]].unique())
                        
                        # run algorithms with h hidden layer and store the evaluated performance.
                        for i in range(len(self.layers)):
                                h = self.layers[i]
                                structure = self.get_network_structure(dataset, len_input_neuron, len_output_neuron,  h)
                                
                                predicted, label = self.run_GA(dataset, trainset, testset, isClassification, structure, h)
                                self.performance_measure(predicted, label, dataset, isClassification, h, self.algorithms)
                                 
                return self.allresults
                        
        def testtrainsplit(self, data, foldlen):
                data = data.sample(frac=1)                   #randomize the rows to avoid sorted data
                testlen = int(len(data) * foldlen)           #split according to fold lenth
                testset = data[testlen:]
                trainset = data[:testlen]
                return trainset, testset
        
        def run_GA(self, key, train, test, isClassification, structure, num_hidden_layers):
                # train and plot convergence
                GA = GeneticAlg.GeneticAlg(train, isClassification, structure)
                epoc, result = GA.train()
                plot_graph(key + ' with ' + str(num_hidden_layers) + ' hidden layers', epoc, result)
                
                # drop class variable
                testClass = test[test.columns[-1]]
                testSet = test.drop([test.columns[-1]], axis = 'columns')
                
                # get predictions
                predicted = GA.test(testSet, np.array(testClass))
                return predicted, testClass
        
        def get_network_structure(self, key, input_neuron, output_neuron, num_hidden_layers):
                if num_hidden_layers == 0:
                        structure = [input_neuron, output_neuron]
                elif num_hidden_layers == 1:
                        hidden_layer = ld.LoadDataset().get1sthiddenlayernode(key)
                        structure = [input_neuron, hidden_layer, output_neuron]
                else:
                        hidden_layer_list = ld.LoadDataset().get2ndhiddenlayernode(key)
                        structure = [input_neuron, hidden_layer_list[0], hidden_layer_list[1], output_neuron]
                return structure
                        
        def performance_measure(self, predicted, labels, dataset, isClassification, h, method):
                #evaluate confusion metrix or root mean square error based on dtaset
                mtrx = Metrics.Metrics()
                if (isClassification):
                        acc, prec, recall = mtrx.confusion_matrix(labels.values, predicted)
                        self.update_result(dataset, isClassification, h, method, acc, prec, recall, 0)
                         
                else:
                        rmse = mtrx.RootMeanSquareError(labels.values, predicted)
                        self.update_result(dataset, isClassification, h, method, 0, 0, 0, rmse)
        
        def update_result(self, dataset, isClassification, h, method, acc, prec, recall, rmse):
                #store result in a dataframe.
                self.allresults = self.allresults.append({'dataset': dataset, 'isClassification': isClassification,
                                                'hiddenLayers': h, 'method': method, 'accuracy': acc, 'precision': prec,
                                                'recall': recall, 'NRMSE': rmse}, ignore_index=True)
        
results = main().main()
results.to_csv('results.csv')
                