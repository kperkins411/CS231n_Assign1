import numpy as np
import matplotlib.pyplot as plt
import math as mth
from classifiers.softmax import *

relu = lambda x: x * (x >= 0).astype(float)
class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  D, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4, use_smart_w1_init = True):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    - std - way to scale weights the default KILLED learning on the HOG and color histogram
            classifier for features.ipynb, they were so small that few changes ever occurred
            So adjust up in general (.01 worked) 
    """
    self.params = {}
    if use_smart_w1_init == True:
        #reLU divide by sqrt(2/fan_in)  works well but kills structure in visualized W1 matrix
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)/mth.sqrt(2.0/input_size)
        
        #divide by sqrt(fan_in) for non reLu  works well but kills structure in visualized W1 matrix
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)/mth.sqrt(hidden_size)
    else: 
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)   #orig use this to get same answer as ipynb 
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)   #orig use this to get same answer as ipynb

    self.params['b1'] = np.zeros(hidden_size)   
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0, dropout_prob = 1.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    num_train, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    #try this way, saves creating vecclamp time and mem
    #compute input*hidden layer
  
    z1 = X.dot(W1) + b1.T
    H1 = relu(z1)

    #lets apply dropout (if ==1 then no change)
    if dropout_prob < 1.0:
        #p=1     #no dropout validation accuracy (val) = .379
        #p=.95  #val = .381
        #p=.92  #val = .381
        #p=.9   #val = .38
        #p=.5    #val = .363
        #p=.25   #val = 
        
        #U1 = np.random.binomial([np.ones(*H1.shape)],.5)* (1.0/(1-0.5))  # does not work
        U1 = (np.random.rand(*H1.shape) < dropout_prob)/dropout_prob
        H1 *=U1
    
    #now this mult the H1 (N,H) matrix by W2 (H,C) to get (N,C)
    #now get the raw scores
    scoresUnNorm=np.dot(H1,W2)+b2.T

    # If the targets are not given then jump out, we're done
    if y is None:
      return scoresUnNorm

    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################

    #compute output of the softmax
    scores = softmax_get_normalized_scores(scoresUnNorm)

    #compute the loss
    loss = softmax_get_data_loss(scores, num_train, y,reg,W1, W2)

    #import pdb; pdb.set_trace()
    
    #compute the gradient for softmax
    dscores =  get_dScores(scores, num_train,y)

    dW2= np.dot(H1.T, dscores)
    dW2 += reg * W2   #d( (1/2)*reg*W**2)/dw = reg*W 
    db2 = np.sum(dscores, axis=0, keepdims=True)

    #get hidden layer gradient, d (3x)/dx = 3, if relu >0 then multiply 1 by coefficient
    dH1 = np.dot(dscores,W2.T) * ((z1 >= 0).astype(float))
    
    #dont count dropped neuron contributions (if ==1 then no change)
    if dropout_prob < 1.0:
        dH1 *= U1
    
    dW1 = np.dot(X.T, dH1)
    dW1 += reg * W1
    #dW1 /=num_train

    # print ("neural_net1 causes -7.30238520e-02")
    # print (dW1)
    db1 = np.sum(dH1, axis=0, keepdims=True)

    # store gradients
    grads = {}
    grads["W1"] = dW1
    grads["W2"] = dW2
    grads["b1"] = db1
    grads["b2"] = db2
    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False, dropout_prob = 1):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      #notice I am opting for replacement here (random choice but use once)
      indices = np.random.choice(num_train,batch_size, replace = False)
      X_batch = X[indices]
      y_batch = y[indices]
        
      #print "x_batch size " + str(X_batch.shape)
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg, dropout_prob=dropout_prob)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W2'] += -learning_rate*grads["W2"]
      self.params['b2'] += -learning_rate*np.squeeze(grads["b2"])

      self.params['W1'] += -learning_rate*grads["W1"]
      self.params['b1'] += -learning_rate*np.squeeze(grads["b1"])
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print ('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    H1=relu(np.dot(X,self.params['W1'])+self.params['b1'].T)
    scoresUnNorm=np.dot(H1,self.params['W2'])+self.params['b2'].T
    y_pred =  np.argmax(scoresUnNorm, axis=1)

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred

