import data
import numpy as np
import matplotlib.pyplot as plt

label_size = 20
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'], mpl.rcParams['ytick.labelsize'] = label_size , label_size

from chainer import FunctionSet, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L

import time
mnist = data.load_mnist_data()

plt.figure(figsize=(10,4))
for i in range(10):
    example = mnist['data'][i].reshape(28, 28)
    target = mnist['target'][i]
    plt.subplot(2, 5, i+1)
    plt.imshow(example, cmap='gray')
    plt.title("Target Number: {0}".format(target))
    plt.axis("off")
plt.tight_layout()
#plt.show()
# Separate the two parts of the MNIST dataset
features = mnist['data'].astype(np.float32) / 255
targets = mnist['target'].astype(np.int32)

# Make a train/test split.
x_train, x_test = np.split(features, [60000])
y_train, y_test = np.split(targets, [60000])
start=time.clock()
# Declare the model layers together as a FunctionSet
mnist_model = FunctionSet(
                                  linear1=L.Linear(784, 300),
                                  linear2=L.Linear(300, 100),
                                  linear3=L.Linear(100, 10)
                         )

# Instantiate an optimizer (you should probably use an
# Adam optimizer here for best performance)
# and then setup the optimizer on the FunctionSet.
mnist_optimizer = optimizers.Adam()
mnist_optimizer.setup(mnist_model)


# Construct a forward pass through the network,
# moving sequentially through a layer then activation function
# as stated above.
def mnist_forward(data, model):
    out1 = model.linear1(data)
    out2 = F.relu(out1)
    out3 = model.linear2(out2)
    out4 = F.relu(out3)
    final = model.linear3(out4)
    return final


# Make a training function which takes in training data and targets
# as an input.
def mnist_train(x, y, model, batchsize=1000, n_epochs=50):
    data_size = x.shape[0]
    # loop over epochs
    for epoch in range(n_epochs):
        print('epoch %d' % (epoch + 1))

        # randomly shuffle the indices of the training data
        shuffler = np.random.permutation(data_size)

        # loop over batches
        for i in range(0, data_size, batchsize):
            x_var = Variable(x[shuffler[i: i + batchsize]])
            y_var = Variable(y[shuffler[i: i + batchsize]])

            output = mnist_forward(x_var, model)
            model.zerograds()
            loss = F.softmax_cross_entropy(output, y_var)
            loss.backward()
            mnist_optimizer.update()


# Make a prediction function, using a softmax and argmax in order to
# match the target space so that we can validate.
def mnist_predict(x, model):
    x = Variable(x)

    output = mnist_forward(x, model)

    return F.softmax(output).data.argmax(1)


mnist_train(x_train, y_train, mnist_model, n_epochs=5)
# Call your prediction function on the test set
pred = mnist_predict(x_test, mnist_model)

# Compare the prediction to the ground truth target values.
accuracy = (pred==y_test).mean()
end=time.clock()

# Print out test accuracy
print("Test accuracy: %f" % accuracy)
print("Cost time: {0} seconds".format(end-start))
serializers.save_hdf5('test.model', mnist_model)
serializers.save_hdf5('test.state', mnist_optimizer)

