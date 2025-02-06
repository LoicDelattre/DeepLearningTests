import numpy as np

'''
N1--\    node1
     \  
       -----SUM--------ACTIVATION--------OUTPUT
     /    weights       sigmoid
N2--/      bias
'''


class MyFirstNeuralNetwork:
    ##FOR A SET AMOUNT OF INPUTS##
    ### 1 layer changeable neurons num in layer##
    def __init__(self, learning_rate, number_of_neurons, number_of_inputs):
        self.weights = []
        self.biases = []
        for j in range(number_of_neurons):
            local_weights = []
            self.biases.append(np.random.randn()) 
            for k in range(number_of_inputs):
                local_weights.append(np.random.randn()) #j neurons, k weights based from input
            self.weights.append(local_weights)
        self.weights = np.array(self.weights)

        
        self.learning_rate = learning_rate
        self.nNeuron = number_of_neurons
        
    def getWeights(self):
        return self.weights
    
    def getBiases(self):
        return self.biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoidDeriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def predict(self, inputVector):
        ##1 output##
        out = 0
        for i in range(0, self.nNeuron):
            layer1 = np.dot(inputVector, self.weights[i]) + self.biases[i] ##dot products acts as linear combination a*x+b*y
            out += self.sigmoid(layer1)
            
        return round(out, 10)
    
    def computeErrorGradient(self, inputVector, target, neuron_id):
        a = inputVector[0]
        b = inputVector[1]
        
        x = self.weights[neuron_id][0] ##1 layer
        y = self.weights[neuron_id][1] ##1 layer
        z = self.biases[neuron_id]

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
    
    def updateParameters(self, derror_dbias, derror_dweights, neuron_id):
        self.biases[neuron_id] = self.biases[neuron_id] - (derror_dbias * self.learning_rate)
        
        self.weights[neuron_id] = self.weights[neuron_id]  - (derror_dweights * self.learning_rate)
        
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

        return cumulativeError/j

    def train(self, inputVectors, targets, iterations, testInput, testTarget):
        cumulativeErrors = []
        testErrors = []
        qt = iterations/100
        for i in range(iterations):
            randDataIndex = np.random.randint(len(inputVectors))

            inputVector = inputVectors[randDataIndex]
            target = targets[randDataIndex]

            for j in range(0, self.nNeuron):
                derror_dbias, derror_dweights = self.computeErrorGradient(inputVector, target, j)
                self.updateParameters(derror_dbias, derror_dweights, j)

            # Measure the cumulative error for all the instances, taken every iterations
            if i % qt == 0:
                cumulativeError = self.sampleErrors(inputVectors, targets)
                cumulativeErrors.append(cumulativeError)
                testErrors.append(self.sampleErrors(testInput, testTarget))

        return cumulativeErrors, testErrors

""" inputVectors = np.array(
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
iterations = 1000
neural_network = MyFirstNeuralNetwork(learning_rate, 1, 2)

trainingError = neural_network.train(inputVectors, targets, iterations)
print(neural_network.getWeights())
print(neural_network.getBiases()) """