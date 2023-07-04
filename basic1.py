import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def makePrediction(inputVector, weights, bias):
    layer1 = np.dot(inputVector, weights) + bias ##dot products acts as linear combination a*x+b*y
    layer2 = sigmoid(layer1)
    return layer2

def derror_weightsFind(inputVector, weights, bias):
    a = inputVector[0]
    b = inputVector[1]

target = 0
inputVector = [1.66, 1.56]
weights1 = [1.45, -0.66]
bias = 0.0

prediction = makePrediction(inputVector, weights1, bias)
print(f"The prediction result is: {prediction}")

