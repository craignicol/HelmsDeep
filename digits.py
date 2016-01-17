from backprop import backprop
import pandas as pd
import numpy as np

# The competition datafiles are in the same directory
# Read competition data files:
train = pd.read_csv("./train.csv")
test  = pd.read_csv("./test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

results = train.loc[:, 'label'] / 10
inputs = train.loc[:, 'pixel0':'pixel783'] / 256

print("Index : {0}".format(train.index))
print("Columns : {0}".format(train.columns))
print("Range of values : inputs [{0}:{1}], results [{2}:{3}]".format(np.min(np.min(inputs)), np.max(np.max(inputs)), np.min(results), np.max(results)))

# Need to set max_training_examples to limit memory usage
# Train on each example twice
bp = backprop(inputs.shape[1], 1, 10, hidden_nodes_2 = 10, max_training_examples = 42000, training_cycles = 100)
bp.train(inputs.values, results.values)

test_inputs = test.values / 256
print("Range of test values : {0} - {1}".format(np.min(np.min(test_inputs)), np.max(np.max(test_inputs))))
test_outputs = bp.classify(test_inputs)

print("Range of {0} results = [{1}:{2}]".format(len(test_outputs), np.min(test_outputs), np.max(test_outputs)))
print(test_outputs)