import numpy as np
np.random.seed(10)

x=np.random.randn(3,4)
class HiddenLayer:
    def __init__(self, inputs_length, num_neurons):
        self.inputs_length= inputs_length
        self.num_neurons = num_neurons
        self.weights = np.random.randn(inputs_length, num_neurons)
        self.biases = np.random.rand(1, num_neurons)

    def forward_pass(self,inputs):
        self.outputs = np.dot(inputs,self.weights) + self.biases
        return self.outputs


layer1=HiddenLayer(4,3)
layer2=HiddenLayer(3,2)

y=layer1.forward_pass(x)
print("layer 1 outputs:",layer1.outputs)

z=layer2.forward_pass(y)
print("layer 2 outputs",layer2.outputs)


