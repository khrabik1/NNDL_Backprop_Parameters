# -*- coding: utf-8 -*-
"""
*******************************
KATHERYN HRABIK
CSC 578 - 710 (ONLINE)
HW2 - - - Network 2
*******************************
"""

##Importing NN578

import NN578_network2 as network2
import numpy as np

# Load the iris train-test (separate) data files
def my_load_csv(fname, no_trainfeatures, no_testfeatures):
    ret = np.genfromtxt(fname, delimiter=',')
    data = np.array([(entry[:no_trainfeatures],entry[no_trainfeatures:]) for entry in ret])
    temp_inputs = [np.reshape(x, (no_trainfeatures, 1)) for x in data[:,0]]
    temp_results = [np.reshape(y, (no_testfeatures, 1)) for y in data[:,1]]
    dataset = list(zip(temp_inputs, temp_results))
    return dataset



iris_train = my_load_csv('D:\\Graduate School\\Depaul - Fall 2018\\Neural Networks and Deep Learning\\Homework\\Hw2\\iris-train-1.csv', 4, 3)

iris_test = my_load_csv('D:\\Graduate School\\Depaul - Fall 2018\\Neural Networks and Deep Learning\\Homework\\Hw2\\iris-test-1.csv', 4, 3)

##-----------------------iris-423.dat Experiments------------------------------


##Experiment 1-----------------------------------------------------------------

net2 = network2.load_network("iris-423.dat")

# Set hyper-parameter values individually after the network
net2.set_parameters(cost=network2.QuadraticCost)

net2.SGD(iris_train, 10, 10, 2.0, evaluation_data=iris_test, monitor_evaluation_cost=True, 
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)
##-----------------------------------------------------------------------------

##Experiment 2-----------------------------------------------------------------

net3 = network2.load_network("iris-423.dat")

# Set hyper-parameter values individually after the network
net3.set_parameters(cost=network2.CrossEntropyCost)

net3.SGD(iris_train, 10, 5, .35, evaluation_data=iris_test, monitor_evaluation_cost=True, 
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)
##-----------------------------------------------------------------------------

##Experiment 3-----------------------------------------------------------------

net6 = network2.load_network("iris-423.dat")

# Set hyper-parameter values individually after the network
net6.set_parameters(cost=network2.CrossEntropyCost, act_output=network2.Softmax, act_hidden=network2.ReLU)

net6.SGD(iris_train, 10, 5, .001, evaluation_data=iris_test, monitor_evaluation_cost=True, 
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)


##-----------------------------------------------------------------------------

##Experiment 4-----------------------------------------------------------------

net1 = network2.load_network("iris-423.dat")

# Set hyper-parameter values individually after the network
net1.set_parameters(cost=network2.LogLikelihood, act_output=network2.Softmax, act_hidden=network2.ReLU)

net1.SGD(iris_train, 10, 10, .001, evaluation_data=iris_test, monitor_evaluation_cost=True, 
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)


##-----------------------------------------------------------------------------

##Experiment 5-----------------------------------------------------------------

net2 = network2.load_network("iris-423.dat")

# Set hyper-parameter values individually after the network
net2.set_parameters(cost=network2.CrossEntropyCost, act_output=network2.Tanh, act_hidden=network2.Tanh)

net2.SGD(iris_train, 10, 3, .25, evaluation_data=iris_test, monitor_evaluation_cost=True, 
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)

##-----------------------------------------------------------------------------

##------------------**Note: experiment 6 was duplicate, removed----------------

##Experiment 7-----------------------------------------------------------------

net1 = network2.load_network("iris-423.dat")

# Set hyper-parameter values individually after the network
net1.set_parameters(cost=network2.LogLikelihood, act_hidden=network2.ReLU, 
                    act_output=network2.Softmax, regularization='L2')

net1.SGD(iris_train, 10, 10, .001, lmbda = 3.0,
            evaluation_data=iris_test, monitor_evaluation_cost=True, 
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)


##-----------------------------------------------------------------------------

##Experiment 8-----------------------------------------------------------------

net1 = network2.load_network("iris-423.dat")

# Set hyper-parameter values individually after the network
net1.set_parameters(cost=network2.LogLikelihood, act_hidden=network2.ReLU, 
                    act_output=network2.Softmax, regularization='L1')

net1.SGD(iris_train, 10, 5, .00125, lmbda = 3.0,
            evaluation_data=iris_test, monitor_evaluation_cost=True, 
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)

##-----------------------------------------------------------------------------

##-----------------------iris-4-20-7-3.dat Experiments-------------------------



##Experiment 9-----------------------------------------------------------------

net1 = network2.load_network("iris-4-20-7-3.dat")

# Set hyper-parameter values individually after the network
# Note due to the way I coded this, dropout gets sent through dropoutmask.
net1.set_parameters(cost=network2.LogLikelihood, act_hidden=network2.ReLU, 
                    act_output=network2.Softmax, dropoutpercent = 0.1)

net1.SGD(iris_train, 10, 1, .00025, lmbda = 3.0,
            evaluation_data=iris_test, monitor_evaluation_cost=True, 
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)

##-----------------------------------------------------------------------------

##Experiment 10----------------------------------------------------------------

net1 = network2.load_network("iris-4-20-7-3.dat")

# Set hyper-parameter values individually after the network
# Note due to the way I coded this, dropout gets sent through dropoutmask.
net1.set_parameters(cost=network2.LogLikelihood, act_hidden=network2.ReLU, 
                    act_output=network2.Softmax,  dropoutpercent = 0.5)

net1.SGD(iris_train, 10, 1, .0005, lmbda = 3.0,
            evaluation_data=iris_test, monitor_evaluation_cost=True, 
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)

##-----------------------------------------------------------------------------

##Experiment 11----------------------------------------------------------------

net1 = network2.load_network("iris-4-20-7-3.dat")

# Set hyper-parameter values individually after the network
# Note due to the way I coded this, dropout gets sent through dropoutmask.
net1.set_parameters(cost=network2.LogLikelihood, act_hidden=network2.ReLU, 
                    act_output=network2.Softmax, regularization = 'L2', dropoutpercent = 0.1)

net1.SGD(iris_train, 10, 1, .1, lmbda = 3.0,
            evaluation_data=iris_test, monitor_evaluation_cost=True, 
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)

##-----------------------------------------------------------------------------

##Experiment 12----------------------------------------------------------------


net1 = network2.load_network("iris-4-20-7-3.dat")

# Set hyper-parameter values individually after the network
# Note due to the way I coded this, dropout gets sent through dropoutmask.
net1.set_parameters(cost=network2.LogLikelihood, act_hidden=network2.ReLU, 
                    act_output=network2.Softmax, regularization = 'L2', dropoutpercent = 0.5)

net1.SGD(iris_train, 10, 1, .1, lmbda = 3.0,
            evaluation_data=iris_test, monitor_evaluation_cost=True, 
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)
