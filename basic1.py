import numpy as np
import matplotlib.pyplot as plt

class MyFirstNeuralNetwork:
    ##PURE SIGMOID NODES##
    ##OUTPUT of network is one node of dot produt of last neruon layer of output and weights##
    def __init__(self, learning_rate, number_of_layers, number_of_neurons_list, number_of_inputs):
        self.weights = []
        self.bias = []
        
        for i in range(number_of_layers+1):
            localLayer = []
            self.bias.append([])
            if i < number_of_layers:
                for j in range(number_of_neurons_list[i-1]):
                    localNeuron = []
                    for k in range(number_of_inputs):
                        localNeuron.append(np.random.randn()) #i layers, j neurons, k weights based from input
                    localLayer.append(localNeuron)
                    self.bias[i].append(np.random.randn())
            else:
                localNeuron = []
                for k in range(number_of_neurons_list[-1]):
                    localNeuron.append(np.random.randn()) #i layers, j neurons, k weights based from input
                localLayer.append(localNeuron)
                self.bias[i].append(np.random.randn())

            self.weights.append(localLayer)

        self.learning_rate = learning_rate
        self.nLayer = number_of_layers
        self.nNeuronList = [number_of_inputs] + number_of_neurons_list
        self.nInputs = number_of_inputs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoidDeriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    #def computePrediciton(self, )
    ##NEED TO CREATE FUNC TO SEPERATELY CALCULATE INPUT AND OUTPUT

    def computeNodesValues(self, inputVector):
        layersOutput = [] 
        for layer in range(self.nLayer):
            layerValues = [[], []] #IN, OUT
            for neuron in range(self.nNeuronList[layer+1]):
                neuronIn = self.bias[layer][neuron]
                if layer == 0:
                    for k in range(len(inputVector)):
                        neuronIn = neuronIn + inputVector[k]*self.weights[layer][neuron][k]
                else:
                    for k in range(len(layersOutput[layer-1][1])):
                        neuronIn = neuronIn + layersOutput[layer-1][1][k]*self.weights[layer][neuron][k]

                neuronOut = self.sigmoid(neuronIn)
                layerValues[0].append(neuronOut)
                layerValues[1].append(neuronIn)
            layersOutput.append(layerValues)

        prediciton = self.bias[layer+1][0]
        for i in range(len(layersOutput[layer-1][1])):
            prediciton = prediciton + layersOutput[layer-1][1][i]*self.weights[layer+1][0][i]
        layersOutput.append(prediciton)


        return layersOutput # 1 list per neuron with dot product val and sigmoid val, last list is just a val of prediciton
    
    def computeWeightGradient(self, derror_dweightList, layer, neuron, derror_dinput, inputValue, dinputNext_dlayer):
        for path in range(self.nNeuronList[layer]):
            dinput_dweight = self.weights[layer][neuron][path] #get previous path weight
            dlayer_dinput = self.sigmoidDeriv(inputValue)
            derror_dweight = derror_dinput*dinputNext_dlayer*dlayer_dinput*dinput_dweight #chain rule
            derror_dweightList[layer][neuron].append(derror_dweight)
        return derror_dweightList
    
    def computeBiasGradient(self, derror_dbiasList, layer, neuron, derror_dprediction, dprediction_dlayer):
        dlayer_dbias = self.bias[layer][neuron]
        derror_dbias = derror_dprediction*dprediction_dlayer*dlayer_dbias
        derror_dbiasList[layer][neuron].append(derror_dbias) 

        return derror_dbiasList

    def dotProduct(self, v1, v2):
        result = 0
        if len(v1) == len(v2):
            for i in range(len(v1)):
                result = result + v1[i]*v2[i]
        else:
            print("ERROR: can't compute dot product of 2 vectors that don't have the same length")
        return result

    def computeErrorGradient(self, inputVector, target):
        ''' derror_dbias, derror_dweights = computeErrorGradient(inputVector, target)
        INPUTS:
        inputVector: numpy array of n elements
        target: int
        '''
        derror_dbiasList = []
        derror_dweightList = []

        nodeValues = self.computeNodesValues(inputVector)
        lastLayerOutput = nodeValues[-2][1]
        outputWeights = self.weights[-1][0] #equivalent to dprediction_dweightOut

        derror_dprediction = 2*(self.dotProduct(lastLayerOutput, outputWeights)  - target)

        derror_dweightList.insert(0, [])
        for i in range(len(outputWeights)):
            derror_dweightList[0].append(outputWeights[i]*derror_dprediction)
            derror_dinput = derror_dprediction

        for i in range(self.nLayer):
            derror_dweightList.insert(0, [])
            derror_dbiasList.insert(0, [])
            layer = self.nLayer-i-1 #if 0 then is layer 1, etc
            for neuron in range(self.nNeuronList[layer+1]):
                derror_dweightList[layer].insert(0, [])
                derror_dbiasList[layer].insert(0, [])
                inputValue = nodeValues[layer][0][neuron]
                dinputNext_dlayer = 
                derror_dweightList = self.computeWeightGradient(derror_dweightList, layer, neuron, derror_dinput, inputValue, dinputNext_dlayer)
                derror_dbiasList = self.computeBiasGradient(derror_dbiasList, layer, neuron, derror_dprediction, 1)

        print(derror_dweightList)
        return derror_dbiasList, derror_dweightList       
    
    def predict(self, inputVector):
        layersValues = self.computeNodesValues(inputVector)
        return layersValues[-1]

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

            derror_dbias, derror_dweights = self.computeErrorGradient(inputVector, target)
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

neural_network = MyFirstNeuralNetwork(learning_rate, 1, [1], 2)

trainingError = neural_network.train(inputVectors, targets, 100)

plt.plot(trainingError)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")