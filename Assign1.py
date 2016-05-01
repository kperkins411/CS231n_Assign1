import numpy as np

__author__ = 'Perkins'
import matplotlib.pyplot as plt
#######################################
W=np.array([[1, 2, 1],
            [9, 3, 4],
            [3, 7, 5]])
X=np.array([[1],[2],[3]])
b=np.array([[.05],[.05],[.05]])
H1 = np.maximum(0, np.dot(W, X) + b)
p=.5

U1 = np.random.rand(*H1.shape) < p
u2= np.random.rand(3,1) < p

H1 *=U1

class SubClass(object):
    def reset(self):
        self.W = None

    def __init__(self):
        self.reset()

    def changeval(self, val):
        self.val = val
    def loss(self):
        pass

class MyClass(SubClass):
    def loss(self):
        print("in loss")

bestval = MyClass()
bestval.changeval(0)
for i in range(3):
    myc = MyClass()
    myc.changeval(i)
    if myc.val > bestval.val:
        bestval = myc


a=np.array([[1, 2, 1],
            [9, 3, 4],
            [3, 7, 5]])
y_pred = np.zeros(a.shape[1])
y_pred = np.argmax(a,axis=1)
d = np.sum(a,axis=0)

b=np.array([1,2,3])
c=a-b
e=b.reshape(1,3)
e=np.transpose(e)
d=a-e


def testfileio():
    x=np.array
    x = numpy.fromfile(file=open("data"), dtype=float).reshape((100, 100, 100))
def comprehension_test():
    matrix = [[1,2,3],[4,5,6],[7,8,9]]
    flat = [[x for x in row] for row in matrix ]
    matrix2= np.array(matrix)
    matrix3= matrix2.reshape(9,)
    matrix4= matrix2.reshape(9,1)
    return flat
comprehension_test()
def transpose_test():
    matrix = [[1,2,3],[4,5,6],[7,8,9]]
    trans = [[row[i] for row in matrix] for i in range(len(matrix))]
    return trans
transpose_test()

matrix = [[1,2,3],[4,5,6],[7,8,9]]

matrix = list(zip(*matrix))  #unpack arguments for access


def test_out():
    num_folds = 2
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    k_all_folds_accuracies = {}

    for i in range(num_folds):
        k_1_fold_accuracy= np.zeros(len(k_choices))

        for k in range(len(k_choices)):
            k_1_fold_accuracy[k] = i*10 +k
        k_all_folds_accuracies[i]= k_1_fold_accuracy

    for k in sorted(k_all_folds_accuracies):
        for idx,accuracy in enumerate(k_all_folds_accuracies[k]):
            print ('k = %d, accuracy = %f' % (k_choices[idx], accuracy))

    #lets convert the useless dict to a matrix and transpose it while we are at it
    k_all_folds_matrix = np.zeros((len(k_choices),num_folds))
    for idxr, row in k_all_folds_accuracies.items():
       for idxc, value in enumerate(row):
           k_all_folds_matrix[idxc, idxr] = value

    # plot the raw observations
    for idx in range(len(k_all_folds_accuracies)):
        accuracies = k_all_folds_accuracies[idx]
        plt.scatter(k_choices, accuracies)
    # plot the trend line with error bars that correspond to standard deviation
    accuracies_mean = np.array([np.mean(v) for v in k_all_folds_matrix])
    accuracies_std  = np.array([ np.std(v) for v in k_all_folds_matrix])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()
test_out()

def setup():
    X_train = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3],[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]])
    X_test =  np.array([[8, 8, 8], [9, 9, 9]])

    y_train = np.array([x for x in range(8)])
    y_test  = np.array([x+8 for x in range(2)])

    print (X_train.shape)
    print (X_test.shape)
    print (y_train.shape)
    print (y_test.shape)
    return (X_train,X_test, y_train, y_test)



def compute_distances_two_loops(train, test):
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
    num_test = test.shape[0]
    num_train = train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
            dists[i, j] = np.sqrt(np.sum(np.square(test[i, :] - train[j, :])))

            # dists = np.sqrt(np.sum(np.square(test - train[j,:]), axis = 1))

    return dists
def compute_distances_one_loop(train, test):
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
    num_test = test.shape[0]
    num_train = train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        q= (train - test[i])
        qq=np.square(q)
        qqq= np.sum(qq, axis =1)
        qqqq=np.sqrt(qqq)
        dists[i] = np.sqrt(np.sum(np.square(train - test[i]),axis =1))
    return dists
def test():
    A = np.ones((3,3,2))
    AA = np.ones((3,3,1))
    AAA = np.ones((3,3))
    v = np.array([1,2])
    c= A.dot(v)
    e= AA.dot(v)
    c= AAA.dot(v)
#test()

def compute_distances_no_loop():
    X_train,X_test, y_train, y_test = setup()

    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    dists = np.zeros((num_test, num_train))

    # from scipy.spatial.distance import cdist
    # dists = cdist(X, self.X_train, metric='euclidean')

    # to fully vectorize, use of the formula: (a-b)^2 = a^2 + b^2 -2ab
    # (a-b)^2 = quadra -2 * prod
    # with quadra = a^2 + b^2; and prod = ab
    a2 = np.sum(X_test ** 2, axis=1) # shape: (2,)
    b2 = np.sum(X_train ** 2, axis=1) # shape: (8,)

    aa2 = a2.reshape(a2.shape[0], 1) # reshape a2 to (2,1) to be able to broadcast a2 and sum to b2

    quadra = aa2 + b2 # shape = (2,8)

    prod = np.dot(X_test, X_train.T) # shape = (2,8)
    dists = np.sqrt(quadra -2 * prod) # shape = (2,8)
    return dists
compute_distances_no_loop()

def broadcastTest2():
    ktrain = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3],[4,4,4],[5,5,5]])
    ktest = np.array([[6, 6, 6], [7, 7, 7], [8, 8, 8]])

    a2 = np.sum(ktest ** 2, axis=1) # shape: (500,)
    b2 = np.sum(ktrain ** 2, axis=1) # shape: (5000,)
    aa2 = a2.reshape(a2.shape[0], 1) # reshape a2 to (500,1) to be able to broadcast a2 and sum to b2
    quadra = aa2 + b2 # shape = (500, 5000)
    prod = np.dot(ktest, ktrain.T) # shape = (500, 5000)
    dists = np.sqrt(quadra -2 * prod) # shape = (500, 5000)
    return dists

def broadcastTest():
    x = np.arange(4)
    xx = x.reshape(4,1)
    xxx= x + xx
    xxxx= x*xx
    y = np.ones(5)
    z = np.ones((3,4))
    x.shape
    q= (xx+y)
    qq=x+1
    zz= x+2
    return zz


def kFoldCrossVal():
    X_train,X_test, y_train, y_test = setup()

    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    X_train_folds = []
    y_train_folds = []
    ################################################################################
    # TODO:                                                                        #
    # Split up the training data into folds. After splitting, X_train_folds and    #
    # y_train_folds should each be lists of length num_folds, where                #
    # y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
    # Hint: Look up the numpy array_split function.                                #
    ################################################################################
    X_train_folds = np.append(X_train,X_test, axis =0)
    y_train_folds = np.append(y_train, y_test,axis = 0)

    # As a sanity check, we print out the size of the training and test data.
    print ('All  data shape: ', X_train_folds.shape)
    print ('All  labels shape: ',y_train_folds.shape)

    X_train_folds= np.array_split(X_train_folds,num_folds)
    y_train_folds= np.array_split(y_train_folds,num_folds)
    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################

    # A dictionary holding the accuracies for different values of k that we find
    # when running cross-validation. After running cross-validation,
    # k_to_accuracies[k] should be a list of length num_folds giving the different
    # accuracy values that we found when using that value of k.

    k_all_folds_accuracies = {}


    ################################################################################
    # TODO:                                                                        #
    # Perform k-fold cross validation to find the best value of k. For each        #
    # possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
    # where in each case you use all but one of the folds as training data and the #
    # last fold as a validation set. Store the accuracies for all fold and all     #
    # values of k in the k_to_accuracies dictionary.                               #
    ################################################################################


    for i in range(num_folds):

        #split out train and test set
        fld_X_test = X_train_folds[i]
        fld_X_train = np.delete(X_train_folds,i,0)         #remove fld_test

        #same for label data
        fld_y_test = y_train_folds[i]
        fld_y_train = np.delete(y_train_folds,i,0)         #remove fld_test

        num_test = len(fld_X_test)

        k_1_fold_accuracy= np.ones(len(k_choices))

        #now lets try different values of k
        for k in range(len(k_choices)):
            # classifier.train(fld_X_train, fld_y_train)
            # dists = classifier.compute_distances_no_loop(fld_y_test)
            # y_test_pred = classifier.predict_labels(dists, k=k_choices[k])

            num_correct = 0
            # num_correct = np.sum(y_test_pred == fld_y_test)
            k_1_fold_accuracy[k] = float(num_correct) / num_test

        #add the accuracy array to
        k_all_folds_accuracies[i]= k_1_fold_accuracy

    ################################################################################
    #                                 END OF YOUR CODE                             #
    ################################################################################

    # Print out the computed accuracies
    # for k in sorted(k_all_folds_accuracies):
    #     for accuracy in k_all_folds_accuracies[k]:
    #         print 'k = %d, accuracy = %f' % (k, accuracy)

kFoldCrossVal()

def arraysplit():
    x = np.arange(8.0)
    y = np.array_split(x, 3)
    return x

broadcastTest()
broadcastTest2()
arraysplit()

##########################################################################################################################
ktrain = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3],[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]])
ktest = np.array([[4, 4, 4], [5, 5, 5]])
dists_3 = compute_distances_no_loop(ktrain, ktest)
square = np.square(ktest[0])    #works
square = np.square(ktest[0,:])  #works
square = np.square(ktest[0])- np.square(ktest[1])
dists_1 = compute_distances_one_loop(ktrain, ktest)
dists_2 = compute_distances_two_loops(ktrain, ktest)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists_1 - dists_2, ord='fro')
print ('Difference was: %f' % (difference, ))
if difference < 0.001:
  print ('Good! The distance matrices are the same')
else:
  print ('Uh-oh! The distance matrices are different')

# dists = compute_distances_two_loops(ktrain, ktest)


##########################################################################################################################
def predict_labels(dists, train_label, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
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

        #get the top k hits
        k_sorted_indices_dists = [sorted_indices_dists[x] for x in range(k)]

        # create an array of associated labels
        closest_y = [train_label[x] for x in k_sorted_indices_dists]

        #find the most common labels
        labels, locs =np.unique(closest_y, return_counts=True)

        #lets find the higest count (the [0] takes the first from a tie)
        index = np.argmax(locs)

        #here is our label
        y_pred[i] = labels[index]
    return y_pred



# dists = np.array([[6, 5, 4, 4, 4], [1, 2, 3, 3, 4], [9, 7, 7, 7, 1]])
dists = np.array([[9, 7, 7, 7, 1]])

# array of labels for training data
# train_label = [x*10 for x in range(dists.shape[1])]
train_label = [10,10,20,20,30]

y_pred = predict_labels(dists, train_label, k=3)
print (y_pred)
