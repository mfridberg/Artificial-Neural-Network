import numpy as np
import csv
import matplotlib.pyplot as plt

LEARNING_RATE = 0.1
NODES = 128
EPOCHS = 20

class NeuralNetwork:
    def __init__(self, network_shapes, learning_rate):
        
        # Create shapes of the weights
        w_shapes = [(x, y) for x, y in zip(network_shapes[1:], network_shapes[:-1])]

        # Save all the different layers shapes
        self.shapes = network_shapes

        # Initialize all the weights with a random value between -1 and 1
        self.weights = [np.random.standard_normal(s) for s in w_shapes]

        # Vector for all the biases
        self.biases = [np.zeros((i, 1)) for i in network_shapes[1:]]

        self.learning_rate = learning_rate

    # Sigmoid Activation Function
    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1.0 - x)
        return 1/(1+np.exp(-x))

    # Softmax Activation Function
    def softmax(self, x, derivative=False):
        if derivative:
            return 1
        return np.divide(np.exp(x), (sum([np.exp(i) for i in x])))

    # Propagate forward through the network, calculating outputs from layers
    def propagate_forward(self, data):
       
        outputs = []
        # For each weight and bias in hidden layer run Sigmoid function
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            data = self.sigmoid(np.matmul(weight, data) + bias)
            outputs.append(data)

        # Softmax activation function for the output layer
        data = self.softmax(np.matmul(self.weights[-1], data) + self.biases[-1])        
        outputs.append(data)

        return outputs

    # Propagate backwards through the network, calculate error terms and change weights
    # Calculate gradient of the loss function with respect to each weight by the chain rule
    def backpropagation(self, data, outputs, target):
    
        # Create a target vector filled with 0, except for index with label value    
        target_vector = np.array([1.0 if i == target else 0.0 for i in range(self.shapes[-1])]).reshape(self.shapes[-1], 1)

        errors = []

        # Initiate our errors list with an empty list for each output
        for i in range(len(outputs)):
            errors.append([])

        # Propagate backwards, calculate error terms
        for i in reversed(range(len(outputs))):

            o = outputs[i]

            # If we are at the output layer
            if i == len(outputs) - 1:

                error = np.subtract(target_vector, o)
                # Calculate errors for nodes in output layer
                errors[i] = np.multiply(self.softmax(o, True), error)
            else:
                next_layer = i + 1

                # Calculate errors downstream
                error = np.matmul(self.weights[next_layer].T, errors[next_layer])
                delta = np.multiply(error, self.sigmoid(o,True))
                errors[i] = delta

        # Calculate delta and change the weights accordingly
        for i in reversed(range(len(outputs))):

            # The input is the previous layers output, or the input data at the input layer
            data_in = data.reshape((self.shapes[0], 1)) if i == 0 else outputs[i - 1]

            # Calculate delta for all weights
            delta_w = np.multiply(np.multiply(errors[i], data_in.T), self.learning_rate)
            # Calculate delta for all biases
            delta_b = np.multiply(np.multiply(errors[i], 1), self.learning_rate)

            # Update with the new values
            self.weights[i] = np.add(self.weights[i], delta_w)
            self.biases[i]  = np.add(self.biases[i], delta_b)

    # Train the network using forward- and backpropagation
    def training(self, training_data):
        print("Training initiated")
        accurate_guesses = 0
        data = []
        labels = []

        # Separate label and data (I accidently normalized the labels before, oops...)
        for d in training_data:
            data.append(d[1:])
            labels.append(d[0])

        # np.divide(data,255) normalizes the data
        for (i, (t, label)) in enumerate(zip(np.divide(data,255), labels)):
            arr = np.array(t)
            outputs = self.propagate_forward(arr.reshape(self.shapes[0],1))

            # Get the index of best guess
            guess = np.argmax(outputs[-1])
            
            # Increment accurate_guesses if the output was correct
            if guess == label:
                accurate_guesses = accurate_guesses + 1

            # Backpropagate through the network
            self.backpropagation(arr, outputs, labels[i])

        # Return the accuracy of the network
        return accurate_guesses / len(training_data)

    # Validate the current state of the network, by calculating the overall accuracy
    def validate(self, validation_data):
        print("Validation initiated")
        accurate_guesses = 0
        data = []
        labels = []

        # Separate label and data
        for d in validation_data:
            data.append(d[1:])
            labels.append(d[0])

        # Propagate forward throught the network, no backpropagation
        for (i, (t, label)) in enumerate(zip(np.divide(data,255), labels)):
            arr = np.array(t)
            outputs = self.propagate_forward(arr.reshape(self.shapes[0],1))

            # Get the index of accurate guess
            guess = np.argmax(outputs[-1])
            
            if guess == label:
                accurate_guesses = accurate_guesses + 1

        # Return the accuracy of the network
        return accurate_guesses / len(validation_data)

    # Test the final accuracy of the network
    def test(self, test_data):
        print("Testing initiated")
        accurate_guesses = 0
        data = []
        labels = []

        # Separate the label and data
        for d in test_data:
            data.append(d[1:])
            labels.append(d[0])

        result_deviation = np.zeros(10)
        amount_of_each = np.zeros(10)
        for (i, (t, label)) in enumerate(zip(np.divide(data,255), labels)):
            arr = np.array(t)
            outputs = self.propagate_forward(arr.reshape(self.shapes[0],1))
            #Get the index of accurate guess
            guess = np.argmax(outputs[-1])

            # Save how many of each handwritten digit exist in the test data
            amount_of_each[int(label)] += 1

            # Check if correct guess
            if guess == label:
                result_deviation[guess] += 1
                accurate_guesses = accurate_guesses + 1

        # Print the accuracy of the network
        percentage = np.divide(result_deviation, amount_of_each)
        print(percentage)
        print(accurate_guesses, "out of", len(test_data))
        return accurate_guesses / len(test_data)

# Load all data from the csv file
def loadAllData():
    print('Loading data.')
    data = []
    
    with open('assignment5.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            data.append([float(i) for i in row])
    
    return data


if __name__ == "__main__":
    # Load all the data from the csv file
    all_data = loadAllData()

    # 70 % of all_data for training
    training_data = all_data[0:int(len(all_data)*0.7)]

    # 10 % of all_data for validation
    validation_data = all_data[int(len(all_data)*0.7):int(len(all_data)*0.8)]

    # 20 % of all_data for testing
    test_data = all_data[int(len(all_data)*0.8):len(all_data)]

    # Shapes of network, this can be changed to add more hidden layers
    network_shapes = [784, NODES ,10]
    e = EPOCHS
    lr = LEARNING_RATE

    # Initiate neural network
    nn = NeuralNetwork(network_shapes, LEARNING_RATE)

    for i in range(1, e+1):
        #Change learning rate every tenth epoch
        print('Epoch:', i)
        if i % 10 == 0:
            lr = lr/10

        training_res = nn.training(training_data)
        validation_res = nn.validate(validation_data)
        print('Validation acc:',"{:.2f}".format(validation_res*100),'%')

    test_res = nn.test(test_data)
    print('Testing accuracy:', "{:.2f}".format(test_res*100), '%')




    


