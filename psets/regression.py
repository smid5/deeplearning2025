from tensorflow.keras.datasets import mnist
import numpy as np
import scipy
import matplotlib.pyplot as plt


# Load the MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data to be suitable for a linear regression model
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Choose digit to make the problem easy enough for linear regression
digit = 3
y_train_binary = (y_train == digit).astype(int)
y_test_binary = (y_test == digit).astype(int)

# Make n power of 2 so we can apply Hadamard
n = 2**15
d = 784
A = x_train[:n,:] # n times d feature matrix
b = y_train_binary[:n] # d dimensional target matrix

# A figure showing the MNIST dataset.
# The binary label is 1 if the image depicts digit.
figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = np.random.choice(len(b))
    img, label = A[sample_idx,:], b[sample_idx]
    img = img.reshape(28,28)
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
