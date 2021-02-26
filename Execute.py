from math import exp


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Test making predictions with the network
dataset = [[0.3, 0.9],
           [0.7, 0.8],
           [0.6, 0.8],
           [0.2, 0.1]]
network = [[{'weights': [3.258878294629293, -0.45130843364828765, -0.9500460500056339]},
            {'weights': [-3.361418039583738, 0.2445850537819873, 0.9728283318044239]},
            {'weights': [-5.294872115156635, 0.08079704797528353, 2.110913004577624]}],
           [{'weights': [-2.8218738050905974, 2.2205719737131075, 3.951589035550646, -0.8031239985718434]},
            {'weights': [2.5382138166876302, -2.4587920344490377, -3.8520720612556576, 1.019444052808084]}]]

print("N1\t\t", "N2\t\t", "Classe")
for row in dataset:
    prediction = predict(network, row)
    if prediction == 0:
        pred = "clou"
    else:
        pred = "vis"
    print (row[0],"\t", row[1],"\t", pred)
