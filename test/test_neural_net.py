import numpy as np
import matplotlib.pyplot as plt
from classifiers.neural_net import TwoLayerNet

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x-y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
  np.random.seed(0) #seed tofor repeatability
  return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
  np.random.seed(1)

  X = 10 * np.random.randn(num_inputs, input_size)  #num_inputs seperate examples, each input_size wide
  y = np.array([0, 1, 2, 2, 1])                     #num_inputs correct classes
  return X, y

net = init_toy_model()
X, y = init_toy_data()

scores = net.loss(X)
print ('Your scores:')
print (scores)
print()
print ('correct scores:')
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print (correct_scores)
print()

# The difference should be very small. We get < 1e-7
print ('Difference between your scores and correct scores:')
print (np.sum(np.abs(scores - correct_scores)))

loss, _ = net.loss(X, y, reg=0.1)
correct_loss = 1.30378789133

# should be very small, we get < 1e-12
print ('Difference between your loss and correct loss:')
print (np.sum(np.abs(loss - correct_loss)))

# Backward pass
# Implement the rest of the function. This will compute the gradient of the loss
# with respect to the variables `W1`, `b1`, `W2`, and `b2`. Now that you (hopefully!)
# have a correctly implemented forward pass, you can debug your backward pass using
# a numeric gradient check:

# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.
from gradient_check import eval_numerical_gradient

loss, grads = net.loss_other(X, y, reg=0.1)

# these should all be less than 1e-8 or so
for param_name in grads:
  f = lambda W: net.loss(X, y, reg=0.1)[0]
  param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
  print ('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

# Train the network
# To train the network we will use stochastic gradient descent (SGD), similar to the SVM
# and Softmax classifiers. Look at the function TwoLayerNet.train and fill in the missing
# sections to implement the training procedure. This should be very similar to the training
# procedure you used for the SVM and Softmax classifiers. You will also have to implement
# TwoLayerNet.predict, as the training process periodically performs prediction to keep track
# of accuracy over time while the network trains.
# Once you have implemented the method, run the code below to train a two-layer network on toy data.
#  You should achieve a training loss less than 0.2.
net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=1e-5,
            num_iters=100, batch_size =5, verbose=False)

print ('Final training loss: ', stats['loss_history'][-1])

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()