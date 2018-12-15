# NNDL_Backprop_Parameters
Modifications to Nielsen's Network2.Py 

Homework Assignment 2: Backprop Hyper-Parameters
Depaul University 
CSC 578 Neural Networks and Deep Learning
Professor Noriko Tomuro
Fall 2018


-	Network Code Parameters to Implement

	-	Cost Functions:
		-	QuadraticCost, CrossEntropy, LogLikelihood
	-	Activation Functions:
		-	Sigmoid, Tanh, ReLu, SoftMax
	-	Regularization:
		-	L1, L2

-	Functions to Modify
	
	-	feedforward()
	-	update_mini_batch()
	-	total_cost()
	-	any others necessary to implement dropout
		/*Note dropout needs update*/


-	Notes

	-	Cost
		-	Each cost function must be implemented as a class
		-	The class should have two static functions: fn()
			executes the definition of the function, and derivative()
			executes the function's derivative to compute error during learning.
		-	fn() returns a scalar, while derivative returns a column vector 
			containing the cost derivative for each node in the output layer)
		-	Notes on LogLikelihood:
			-	This cost function should only be used when the activation function
				of the output layer = softmax.

	-	act_hidden
		-	This parameter specifies the activation function for nodes on all HIDDEN
			layers only (excluding output layer)
	
	-	act_output
		-	This parameter specifies the activation function for the nodes on
			the output layer only
		-	Note that, for sigmoid, tanh, and relu, the parameter z could be either
			a vector or a scalar depending on how the code is written. Note that for
			softmax, however, the parameter z must be assumed to be a vector.
		-	Softmax is allowed only for the output layer. Since its derivative returns
			a 2D matrix instead of a vector, it is handled differently when computing
			the error/delta in backprop().


