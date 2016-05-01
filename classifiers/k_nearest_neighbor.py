import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        dists[i,j]= np.sqrt(np.sum(np.square(X[i,:] - self.X_train[j,:])))
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):   
      dists[i] = np.sqrt(np.sum(np.square(self.X_train -X[i]),axis =1))
     
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    # from scipy.spatial.distance import cdist
    # dists = cdist(X, self.X_train, metric='euclidean')

    # to fully vectorize, use of the formula: (a-b)^2 = a^2 + b^2 -2ab
    # (a-b)^2 = quadra -2 * prod
    # with quadra = a^2 + b^2; and prod = ab
    a2 = np.sum(X ** 2, axis=1)                               # shape: (500,)
    b2 = np.sum(self.X_train ** 2, axis=1)                    # shape: (5000,)
    aa2 = a2.reshape(a2.shape[0], 1)                          # reshape a2 to (500,1) to be able to broadcast a2 and sum to b2
    quadra = aa2 + b2                                         # shape = (500, 5000)
    prod = np.dot(X, self.X_train.T)                          # shape = (500, 5000)
    dists = np.sqrt(quadra -2 * prod)                         # shape = (500, 5000)
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      # sort a row of dists, return the indices of where lowest to highest occur
      sorted_indices_dists = np.argsort(dists[i, :])
        
      #import pdb; pdb.set_trace()
 
      #get the top k hits
      k_sorted_indices_dists = [sorted_indices_dists[x] for x in range(k)]
        
      # create an array of associated labels
      closest_y = [self.y_train[x] for x in k_sorted_indices_dists]

      #find the most common labels
      labels, locs =np.unique(closest_y, return_counts=True)

      #lets find the higest count (the [0] takes the first from a tie)
      index = np.argmax(locs)

      #here is our label
      y_pred[i] = labels[index]
      #print "evaluating" + str(i)
    return y_pred

