# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 13:25:31 2019

@author: cnoble

GeneticAlg trains and tests the genetic algorithm. 
initPop() intializes the population of the set number from popSize. It creates
    each individual from the Individual class, and calls setFitness() with the
    training set and training output.
train() runs through the set number of generations to train the algorithm. It
    saves the performance over time to return. It first finds the current best
    and saves it if it's better than previous best. It adds this indidiual to 
    the next generation. It then keeps a random quarter of the current 
    population to carry forward to the next generation. It then hosts 
    tournaments to get children for the next generation. It then calls mutate
    to mutate the population.
selection() randomly selects a set number of individuals from the population
    and calls getMostFit() to get the winner.
getMostFit() returns the winner of a selection based on most fit.
crossover() takes two parents and crosses them using uniform selection with a 
    set percent chance of flipping each gene.
reassignCrossoverGenes() takes two parents, two children, and the selected 
    gene to flip and flips them.
mutation() randomly selects some individuals and calls mutate() from the
    Individual class to mutate their genes.
test() takes the test or validation points and their associated output and 
    uses the most fit individual to call predict() and get fitness.
predict() takes a test set and individual to call feedforward in the Individual
    class to get predictions.
calcAveFitness() loops through the population to monitor average fitness.

"""
import Individual
import random as rn
import numpy as np
from sklearn.model_selection import train_test_split
import copy
import statistics

class GeneticAlg:
        
        def __init__(self, trainSet, isClassification, networkStructure):
            # initialize network structure and layers
            self.classification = isClassification
            self.networkStructure = networkStructure
            self.trainSet, self.valSet = train_test_split(trainSet, test_size = 0.3)
            self.outputLayer = []
            for i in range(networkStructure[-1]):
                self.outputLayer.append(0)
            
            # drop class column
            self.trainClass = np.array(self.trainSet[self.trainSet.columns[-1]])
            self.trainSet = self.trainSet.drop([self.trainSet.columns[-1]], axis = 'columns')
            self.valClass = np.array(self.valSet[self.valSet.columns[-1]])
            self.valSet = self.valSet.drop([self.valSet.columns[-1]], axis = 'columns')
            
            # initilize GA attributes
            self.popSize = 100
            self.population = []
            self.bestIndivdiual = -1
            self.bestIndivdiualFitness = -1
            self.performance = []
            self.initPop()

        # intialize the population of individuals
        def initPop(self):
            for i in range(self.popSize):
                newInd = Individual.Individual(self.networkStructure, self.classification)
                newInd.setFitness(self.trainSet, self.trainClass)
                # find best fit
                if newInd.fitness > self.bestIndivdiualFitness:
                    self.bestIndivdiual = copy.deepcopy(newInd)
                    self.bestIndivdiualFitness = newInd.fitness
                # add individual to population
                self.population.append(newInd)
    
        # train algorithm
        def train(self):
            maxIterations = 40
            generation = 0
            generationList = range(0, maxIterations)
            
            # repeat until max iterations is reached
            while(generation < maxIterations):
                print('generation: ', generation)
                nextGeneration = []
                
                # get current most fit individial
                mostFitIndex = self.getMostFit(range(self.popSize))
                mostFit = self.population[mostFitIndex]
                # keep current most fit in new generation
                nextGeneration.append(mostFit)
                
                # keep random quarter of population in new generation
                randomSelection = rn.sample(self.population, int(self.popSize/4))
                nextGeneration.extend(randomSelection)

                # tournament selection to select parents
                tournamentWinners = []
                # tornaments replace approximately 3/4 of population
                numTournaments = int(3*(self.popSize/4)-1)
                numParents = 2
                # host tournament and get winners to serve as parents
                for i in range(numTournaments):
                    parents = []
                    for j in range(numParents):
                        winner = self.selection()
                        parents.append(winner)
                    tournamentWinners.append(parents)
                # perform crossover and add children to new generation
                for i in range(int(len(tournamentWinners)/2)):
                    children = self.crossover(tournamentWinners[i])
                    nextGeneration.extend(children)
                
                # replace population with new generation
                self.population = nextGeneration
                # apply mutation to population
                self.mutation()

                # monitor most fit training fitness
                print("----training fitness----")
                currentBestFitness = mostFit.fitness
                if not self.classification: currentBestFitness = -currentBestFitness
                print("current best training fitness", round(currentBestFitness, 5))

                # monitor average training fitness
                aveFitness = self.calcAveFitness()
                if not self.classification: aveFitness = -aveFitness
                print("average training fitness", round(aveFitness, 5))
                                
                # monitor validation fitness
                print("----validation fitness----")
                self.test(self.valSet, self.valClass)
                
                generation += 1
            
            return generationList, self.performance
            
        # use tournament style selection to select individuals                
        def selection(self):
            selected = []
            k = 5
            # select k participants
            for i in range(k):
                popIndex = rn.randrange(self.popSize)
                selected.append(popIndex)
            # get winner of tournament
            winner = self.getMostFit(selected)
            return winner
        
        # get fittest of selected individuals
        def getMostFit(self, selected):
            maxFitness = -1
            mostFitIndex = -1
            for i in selected:
                fitness = self.population[i].fitness
                if(fitness > maxFitness):
                    maxFitness = fitness
                    mostFitIndex = i
            return mostFitIndex
        
        # use uniform crossover to combine 2 individuals to create offspring
        def crossover(self, parentsInd):
            # create children based on parents genome 
            parents = []
            children = []
            # create children intially similar to parents
            for i in range(2):
                parents.append(self.population[parentsInd[i]])
                children.append(copy.deepcopy(parents[i]))

            # for network with hidden layers, loop through extra dimension
            if len(self.networkStructure) > 2:
                for outerInd, midd in np.ndenumerate(children[0].chromosome):
                    for innerInd, value in np.ndenumerate(midd):
                        # random x% chance of swapping each gene
                        result = rn.randint(0, 4)
                        if result == 0:
                            i = outerInd[0]
                            j = innerInd[0]
                            k = innerInd[1]
                            children = self.reassignCrossoverGenes(children, parents, i, j, k)
            # for no hidden layers, loop through single dimension
            else:
                for outerInd, value in np.ndenumerate(children[0].chromosome):
                    # random x% chance of swapping each gene
                    result = rn.randint(0, 4)
                    if result == 0:
                        i = outerInd[0]
                        j = outerInd[1]
                        k = outerInd[2]
                        children = self.reassignCrossoverGenes(children, parents, i, j, k)
                        
            # calculate children's fitness
            for i in range(len(children)):
                children[i].setFitness(self.trainSet, self.trainClass)
            return children  
        
        # flip genes for crossover           
        def reassignCrossoverGenes(self, children, parents, i, j, k):
            children[0].chromosome[i][j][k] = parents[1].chromosome[i][j][k]
            children[1].chromosome[i][j][k] = parents[0].chromosome[i][j][k]
            return children
                       
        # randomly select some individuals to mutate genes
        def mutation(self):
            # randomly select k individuals
            k = int(self.popSize/2)
            toMutate = rn.sample(range(self.popSize), k)
            for i in toMutate:
                self.population[i].mutate()
                # recalculate fitness after mutation
                self.population[i].setFitness(self.trainSet, self.trainClass)
        
        # predict based on most fit individual
        def test(self, testSet, testClass):
            
            # test with overall most fit
            bestPredicted = self.predict(testSet, self.bestIndivdiual)
            bestFitness = self.bestIndivdiual.evalFitness(bestPredicted, testClass)
            bestFitnessP = bestFitness
            if not self.classification: bestFitnessP = -bestFitnessP
            print("overall most fit test fitness", round(bestFitnessP, 5))

            # test with current most fit
            mostFitIndex = self.getMostFit(range(self.popSize))
            mostFit = self.population[mostFitIndex]
            predicted = self.predict(testSet, mostFit)
            fitness = mostFit.evalFitness(predicted, testClass)
            fitnessP = fitness
            if not self.classification: fitnessP = -fitnessP
            print("current most fit test fitness", round(fitnessP, 5), "\n")
            # store fitness to return
            self.performance.append(fitnessP)
            
            # if current fitness is better than previous, store this indivdiual
            if fitness > bestFitness:
                self.bestIndivdiual = copy.deepcopy(mostFit)
                self.bestIndivdiualFitness = fitness
                bestPredicted = predicted

            return bestPredicted
        
        # predict based on individual's chromosome
        def predict(self, testSet, individual):
            predictions = []
            # use each test point and the selected individual
            for index, point in testSet.iterrows():
                output = individual.feedforward(point.values)
                predicted = individual.predict(output)
                predictions.append(predicted)
            return predictions
        
        # get the average fitness of the population
        def calcAveFitness(self):
            fitnesses = []
            # get all fitnesses
            for i in range(self.popSize):
                currIndividual = self.population[i]
                fitness = currIndividual.fitness
                fitnesses.append(fitness)
            # average fitnesses
            average = statistics.mean(fitnesses)
            return average
    