# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 13:26:41 2019

@author: cnoble

Inidividual contains the chromosome structure for the individual object, and 
    the fitness of the individual based on the training set.
intializeGenes() intializes weights randomly based on network structure.
setFitness() uses feedforward(), predict(), and evalFitness() to set the
    fitness for the individual.
feedforward() feeds forward through the network structure to get the output 
    layer. It uses the sigmoid from the Metrics class.
predict() predicts the class or regression value based on the output layer.
    It chooses the highest value in the output layer for classification,
    and takes thevalue in the single output node for regression.
evalFitness() calculates a fitness value for the individual using metrics from
    the Metrics class. It uses rsme for regression, and inverts the value to
    make it a maximization problem. For classification, it calculates accuracy.
mutate() randomly selects a portion of genes and mutates them.

"""
import random as rn
import numpy as np
import Metrics as mt

class Individual:
        
        def __init__(self, chromosomeStructure, isClassification):
            self.isClassification = isClassification
            self.chromosomeStructure = chromosomeStructure
            self.chromosome = []
            self.genomeLength = 0
            self.intializeGenes()
            self.fitness = 0
            
        # intialize genes randomly
        def intializeGenes(self):
            # loop through network structure to set chromosome structure
            for i in range(1, len(self.chromosomeStructure)):
                # add an extra weight for bias node
                numRows = self.chromosomeStructure[i-1] + 1
                numCols = self.chromosomeStructure[i]
                # initialize with normal distribution centered at 0
                layerWeights = np.random.normal(0, 0.2, (numRows, numCols))
                self.chromosome.append(layerWeights.tolist())
                length = numRows*numCols
                self.genomeLength += length
            
        # set the individual's fitness
        def setFitness(self, trainSet, trainClass):
            # loop through training set and feedforward to get predictions
            predictions = []
            for index, point in trainSet.iterrows():
                output = self.feedforward(point.values)
                predicted = self.predict(output)
                predictions.append(predicted)
            # evaluate fitness based on predicted and expected class
            self.fitness = self.evalFitness(predictions, trainClass)
        
        # feedforward to get output layer for network
        def feedforward(self, trainPoint):
            current = trainPoint
            # add bias node to initial input
            current = np.append(current, [1])
            # feedforward through network layers
            for i in range(len(self.chromosomeStructure)-1):
                current = np.dot(current, self.chromosome[i])
                current = mt.Metrics().sigmoid(current)
                # add bias node to all layers except output layer
                if i != len(self.chromosomeStructure)-2:
                    current = np.append(current, [1])
            # return output layer
            return current
            
        # get prediction for classification vs regression based on output layer
        def predict(self, output):
            if(self.isClassification):
                predicted = np.argmax(output)
            else:
                predicted = output[0]
            return predicted
        
        # evaluate fitness based on predicted vs. expected
        def evalFitness(self, predictions, expected):
            if self.isClassification:
                # use accuracy as fitness measure
                result = mt.Metrics().confusion_matrix(expected, predictions)
                fitness = result[0]
            else:
                # invert rmse to deal with maximization problem
                result = mt.Metrics().RootMeanSquareError(expected, predictions)
                fitness = -result
            return fitness
        
        # mutate a subset of genes
        def mutate(self):
            # percentage of genes to mutate
            numMutations = int(self.genomeLength/2)
            count = 0
            while count < numMutations:
                # randomly select indices of chromosome matrix to mutate
                i = rn.sample(range(len(self.chromosome)), 1)[0]
                j = rn.sample(range(len(self.chromosome[i])), 1)[0]
                k = rn.sample(range(len(self.chromosome[i][j])), 1)[0]
                # mutate by random amount from normal distribution centered at 0
                mutationRate = rn.gauss(0, 0.05)
                self.chromosome[i][j][k] += mutationRate
                count += 1
                
        def setChoromosome(self, newChoromosome):
                self.chromosome = newChoromosome
 
                