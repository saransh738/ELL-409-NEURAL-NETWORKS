import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import math
from timeit import default_timer as timer

class NEURAL(object):
    def __init__(self,num_inputs,hidden_layers,num_outputs):
        
        # num_inputs = number of inputs
        # hidden_layers = a list of int for hidden layers
        # num_outputs = number of outputs    
        
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        
        layers = [num_inputs] + hidden_layers +[num_outputs]
        
        #initialize random weights matrix
        weights =[] 
        for i in range (len(layers)-1):
            weights.append(np.random.randn(layers[i],layers[i+1])/100)
        self.weights = weights  
         
        activations =[] 
        for i in range (len(layers)):
            t = np.zeros(layers[i])
            activations.append(t)
        self.activations = activations     
          
            
        derivatives =[] 
        for i in range (len(layers)-1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives      
            
            
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))

    def sigmoid_derivative(self,x):
        return x*(1.0-x)

    def tanh(self,x):
        return (2.0/(1.0+np.exp(-2*x)))-1

    def tanh_derivative(self,x):
        return 1.0-x*x

    def forward_propogation(self,inputs):
        activations = inputs
        self.activations[0] = activations

        for i,w in enumerate(self.weights):
        
            net_inputs = np.dot(activations,w)
            activations = self.tanh(net_inputs)
            self.activations[i+1] = activations
        return activations
     
    def back_propagate(self,error):
        
        for i in reversed (range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error*self.tanh_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped,delta_reshaped)
            error = np.dot(delta, self.weights[i].T)
            
    def gradient_descent(self, eta):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * eta
              
    def train(self, inputs, targets, epochs, eta):
        sum_error = 0
        for i in range(epochs):
            for j,input in enumerate(inputs):
                target = [0 for i in range(10)]
                target[int(targets[j])] =1
                output = self.forward_propogation(input)
                error = target - output
                self.back_propagate(error)
                self.gradient_descent(eta)
                sum_error += self.MSE(target, output)
                
            #print("Error:",sum_error/len(inputs),"in epoch:",i)
        
    def MSE(self,target,output):
        return np.average((target-output) ** 2)
        

def prediction(inp):
    o = np.arange(len(inp))
    for i in range(len(inp)):
        t = inp[i]
        r = np.where(t == np.amax(t))
        o[i]=r[0]
    return o

def score(actual, predicted):
    score = 0
    m = len(actual)
    for i in range(len(actual)):
        if (int(actual[i]) == predicted[i]):
            score+=1
    return (score/m)*100

def showImage(image,label):
    fim = np.array(image, dtype='float')
    l = int(math.sqrt(image.size))
    pix = fim.reshape((l,l)).T
    plt.imshow(pix, cmap='gray')
    print (label)
    plt.show()
        
#Reading files
data_points_train = pd.read_csv('2019MT60763.csv', header = None, nrows = 3000)
data = np.array(data_points_train.values)

#training data
train_x = data[:2500,:25]
train_t = data[:2500,25]

#testing data
input = data[500:,:25]
test_t = data[500:,25]


# n = number of neurons in hidden layer
n = 30
M = NEURAL(25, [50,60], 10)
eta = 0.7
epoch = 10
M.train(train_x,train_t, epoch, eta)
print(score(train_t,prediction(M.forward_propogation(train_x))))
print(score(test_t,prediction(M.forward_propogation(input))))



'''matrix = np.zeros((20,3))
for i in range(1,20):
    M = NEURAL(25,[50],10)
    M.train(train_x,train_t, 10 , i/1000)
    matrix[i][0] = i/1000
    matrix[i][1] = score(train_t,prediction(M.forward_propogation(train_x)))
    matrix[i][2] = score(test_t,prediction(M.forward_propogation(input)))
    
fig = plt.figure(1)
plt.plot(matrix[1:,0:1],matrix[1:,1:2],label = 'Training')
plt.plot(matrix[1:,0:1],matrix[1:,2:3],label = 'Testing')
plt.xlabel('eta')
plt.ylabel('Accuracy')
plt.title('Accuracy vs eta')
plt.legend()
plt.show()

matrix = np.zeros((10,3))
for i in range(1,10):
    M = NEURAL(25,[10*i],10)
    M.train(train_x,train_t, 10 , 0.008)
    matrix[i][0] = 10*i
    matrix[i][1] = score(train_t,prediction(M.forward_propogation(train_x)))
    matrix[i][2] = score(test_t,prediction(M.forward_propogation(input)))
 
print(matrix) 
fig = plt.figure(1)
plt.plot(matrix[1:,0:1],matrix[1:,1:2],label = 'Training')
plt.plot(matrix[1:,0:1],matrix[1:,2:3],label = 'Testing')
plt.xlabel('Number of Neuron ')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Neuron')
plt.legend()
plt.show()


matrix = np.zeros((11,3))
for i in range(1,11):
    M = NEURAL(784,[30,60],10)
    M.train(train_x,train_t, 10 , i/1000)
    matrix[i][0] = i/1000
    matrix[i][1] = score(train_t,prediction(M.forward_propogation(train_x)))
    matrix[i][2] = score(test_t,prediction(M.forward_propogation(input)))
    
fig = plt.figure(1)
plt.plot(matrix[1:,0:1],matrix[1:,1:2],label = 'Training')
plt.plot(matrix[1:,0:1],matrix[1:,2:3],label = 'Testing')
plt.xlabel('eta')
plt.ylabel('Accuracy')
plt.title('Accuracy vs eta')
plt.legend()
plt.show()

matrix = np.zeros((20,3))
for i in range(1,20):
    M = NEURAL(784,[30],10)
    M.train(train_x,train_t, i ,0.016)
    matrix[i][0] = i
    matrix[i][1] = score(train_t,prediction(M.forward_propogation(train_x)))
    matrix[i][2] = score(test_t,prediction(M.forward_propogation(input)))
    
fig = plt.figure(3)
plt.plot(matrix[1:,0:1],matrix[1:,1:2],label = 'Training')
plt.plot(matrix[1:,0:1],matrix[1:,2:3],label = 'Testing')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs epoch')
plt.legend()
plt.show()


matrix = np.zeros((20,2))
for i in range(1,20):
    M = NEURAL(25,[50],10)
    M.train(train_x,train_t, 10 , i/1000)
    matrix[i][0] = i/1000
    start = timer()  
    a1 = prediction(M.forward_propogation(train_x))
    a2 = prediction(M.forward_propogation(input))
    end = timer()  
    matrix[i][1] = end-start
    
fig = plt.figure(4)
plt.plot(matrix[:,0],matrix[:,1],label = 'Time')
plt.xlabel('eta')
plt.ylabel('Time')
plt.title('Time vs eta')
plt.legend()
plt.show()

matrix = np.zeros((10,2))
for i in range(1,10):
    M = NEURAL(25,[10*i],10)
    M.train(train_x,train_t, 10 , 0.008)
    matrix[i][0] = 10*i
    start = timer()  
    a1 = prediction(M.forward_propogation(train_x))
    a2 = prediction(M.forward_propogation(input))
    end = timer()  
    matrix[i][1] = end-start
    
fig = plt.figure(3)
plt.plot(matrix[:,0],matrix[:,1],label = 'Time')
plt.xlabel('Number of Neuron')
plt.ylabel('Time')
plt.title('Time vs Number of Neuron')
plt.legend()
plt.show()'''