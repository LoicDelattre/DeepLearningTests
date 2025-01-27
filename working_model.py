import numpy as np
import matplotlib.pyplot as plt

class MyFirstNeuralNetwork:
    ##FOR A SET AMOUNT OF INPUTS##
    def __init__(self, learning_rate, number_of_layers, number_of_neurons_list, number_of_inputs):
        self.weights = []
        for i in range(number_of_layers):
            localWeights = []
            for j in range(number_of_neurons_list[i-1]+1):
                for k in range(number_of_inputs):
                    localWeights.append(np.random.randn()) #i layers, j neurons, k weights based from input
            self.weights.append(localWeights)
        self.weights = np.array(self.weights)

        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.nLayer = number_of_layers
        self.nNeuronList = number_of_neurons_list

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoidDeriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def predict(self, inputVector):
        layer1 = np.dot(inputVector, self.weights[self.nLayer-1]) + self.bias ##dot products acts as linear combination a*x+b*y
        layer2 = self.sigmoid(layer1)
        return layer2
    
    def computeErrorGradient(self, inputVector, target, localLayer):
        a = inputVector[0]
        b = inputVector[1]
       
        x = self.weights[localLayer][0]
        y = self.weights[localLayer][1]
        z = self.bias

        layer1 = a*x+b*y+z
        layer2 = self.sigmoid(layer1)

        dlayer1_dx = a
        dlayer1_dy = b
        dlayer1_dz = 1

        derror_dprediction = 2 * (layer2 - target)
        dprediction_dlayer1 = self.sigmoidDeriv(layer1)
        chainRule = derror_dprediction*dprediction_dlayer1

        derror_dweights = np.array([chainRule*dlayer1_dx, chainRule*dlayer1_dy])
        derror_dbias = chainRule*dlayer1_dz

        return derror_dbias, derror_dweights        
    
    def updateParameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        
        self.weights = self.weights - (derror_dweights * self.learning_rate)
        
        return
    
    def sampleErrors(self, inputVectors, targets):
        cumulativeError = 0
        # Loop through all the instances to measure the error
        for j in range(len(inputVectors)):
            data_point = inputVectors[j]
            target = targets[j]
            prediction = self.predict(data_point)
            error = np.square(prediction - target)

            cumulativeError = cumulativeError + error

        return cumulativeError

    def train(self, inputVectors, targets, iterations):
        cumulativeErrors = []
        for i in range(iterations):
            randDataIndex = np.random.randint(len(inputVectors))

            inputVector = inputVectors[randDataIndex]
            target = targets[randDataIndex]

            for j in range(self.nLayer):
                derror_dbias, derror_dweights = self.computeErrorGradient(inputVector, target, j)
                self.updateParameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances, taken every iterations
            if i % 50 == 0:
                cumulativeError = self.sampleErrors(inputVectors, targets)
                cumulativeErrors.append(cumulativeError)

        return cumulativeErrors


inputVectors = np.array(
    [
        [3, 1.5],
        [2, 1],
        [4, 1.5],
        [3, 4],
        [3.5, 0.5],
        [2, 0.5],
        [5.5, 1],
        [1, 1],
    ]
 )

targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
learning_rate = 0.1

neural_network = MyFirstNeuralNetwork(learning_rate, 1, [1])

trainingError = neural_network.train(inputVectors, targets, 10000)

plt.plot(trainingError)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")