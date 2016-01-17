import numpy as np
from math import sin, cos

class backprop():
    weights1 = []
    weights2 = []
    weights3 = None
    
    training_cycles = 60000
    max_training_examples = 2
    
    def __init__(self, input_size, output_size, hidden_nodes_1, hidden_nodes_2 = 0, seed = 1, training_cycles = 60000, max_training_examples = 0):
        np.random.seed(seed)
        self.training_cycles = training_cycles
        self.max_training_examples = max_training_examples
        
        # Let's define 2 or 3 hidden layers
        # input (X) * weights1 * weights2 (* weights3) = output (Y)
        self.weights1 = 2 * np.random.random((input_size, hidden_nodes_1)) - 1
        if hidden_nodes_2 == 0:
            self.weights2 = 2 * np.random.random((hidden_nodes_1, output_size)) - 1
        else:
            self.weights2 = 2 * np.random.random((hidden_nodes_1, hidden_nodes_2)) - 1
            self.weights3 = 2 * np.random.random((hidden_nodes_2, output_size)) - 1
    
    def nonlin(self,x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))
    
    def propagate(self, X):
        # input (X) * weights1 * weights2 = output (Y)
        # Feed forward through layers 0, 1, and 2
        l0 = X
        l1 = self.nonlin(np.dot(l0,self.weights1))
        l2 = self.nonlin(np.dot(l1,self.weights2))
        if self.weights3 is not None:
            l3 = self.nonlin(np.dot(l2,self.weights3))
            return(l3, l2, l1)
        
        return (l2, l2, l1)
    
    def train_once(self, l0, Y):
        (l3, l2, l1) = self.propagate(l0)
        
        # level 3 learning
        if self.weights3 is not None:
            l3_error = Y - l3
            l3_delta = l3_error * self.nonlin(l3,deriv=True)
            
            l2_error = l3_delta.dot(self.weights3.T)
        else:
            # level 2 learning
            l2_error = Y - l2
            
        l2_delta = l2_error * self.nonlin(l2,deriv=True)
        
        # level 1 learning
        l1_error = l2_delta.dot(self.weights2.T)
        l1_delta = l1_error * self.nonlin(l1,deriv=True)
        
        # update weights
        if self.weights3 is not None:
            self.weights3 += l2.T.dot(l3_delta)
        self.weights2 += l1.T.dot(l2_delta)
        self.weights1 += l0.T.dot(l1_delta)
        
        return l2_error
    
    def train(self, inputs, outputs):
        assert(len(inputs) == len(outputs))
        training_size = len(inputs)
        
        if self.max_training_examples > len(inputs):
            max_training_examples = len(inputs)
        else:
            max_training_examples = self.max_training_examples
            
        for x in range(self.training_cycles):
            start = x % training_size
            end = min(start + max_training_examples, training_size)
            
            input_subset = np.array([inputs[start:end]])
            output_subset = np.array([outputs[start:end]]).T
            error = self.train_once(inputs, outputs)
            if (x % 10000) == 0:
                print ("Error: " + str(np.mean(np.abs(error))))
    
    def classify(self, X):
        return self.propagate(X)[0]

if __name__ == '__main__':
    X = np.array([[sin(x/10) for x in range(1000)],
                [1] * (1000),
                [cos(x/10) for x in range(1000)],
                [0] * (1000)] * 10000)
                
    Y = np.array([[1],
                [0],
                [1],
                [0]] * 10000)
    
    bp = backprop(len(X[0]), len(Y[0]), 4, hidden_nodes_2 = 4, max_training_examples = 10)
    bp.train(X, Y)

    print ("Output (line) - should be 0 : {:1.3f}".format(bp.classify([0.5] * 1000)[0]))
    print ("Output (sawtooth) - should be 0 : {:1.3f}".format(bp.classify([(x / 10.0) % 1 for x in range(1000)])[0]))
    print ("Output (offset sin) - should be 1 : {:1.3f}".format(bp.classify([sin((x+10)/10) for x in range(1000)])[0]))