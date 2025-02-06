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
        for j in range(number_of_neurons):
            local_weights = []
            for k in range(number_of_inputs):
                local_weights.append(np.random.randn()) #j neurons, k weights based from input
            self.weights.append(local_weights)
        self.weights = np.array(self.weights)

        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.nNeuron = number_of_neurons
        
    def getWeights(self):
        return self.weights

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoidDeriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def predict(self, inputVector):
        layer1 = np.dot(inputVector, self.weights[0]) + self.bias ##dot products acts as linear combination a*x+b*y
        layer2 = self.sigmoid(layer1)
        return layer2
    
    def computeErrorGradient(self, inputVector, target):
        a = inputVector[0]
        b = inputVector[1]
        
        x = self.weights[0][0] ##1 neuron
        y = self.weights[0][1] ##1 neuron
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
        
        self.weights[0] = self.weights[0] - (derror_dweights * self.learning_rate)
        
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

    def train(self, inputVectors, targets, iterations):
        cumulativeErrors = []
        qt = iterations/100
        for i in range(iterations):
            randDataIndex = np.random.randint(len(inputVectors))

            inputVector = inputVectors[randDataIndex]
            target = targets[randDataIndex]

            derror_dbias, derror_dweights = self.computeErrorGradient(inputVector, target)
            self.updateParameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances, taken every iterations
            if i % qt == 0:
                cumulativeError = self.sampleErrors(inputVectors, targets)
                cumulativeErrors.append(cumulativeError)

        return cumulativeErrors