import numpy as np
from random import shuffle

#hello
def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    #from http://cs231n.github.io/optimization-1/#opt3
    #the number of times xi is added to gradient at w-sub-yi
    #once for every time expression evaluates above 0
    cnt =0
    for j in range(num_classes):
        if j == y[i]:
            continue

        margin = scores[j] - correct_class_score + 1 # note delta = 1
        if margin > 0:
            loss += margin
            dW[:,j]+=X[i,:]
            cnt+=1
    dW[:,y[i]]-=X[i,:]*cnt

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

 # Also use the averaged gradient of full batch
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # add regularization gradient
  dW += reg * W

  return loss, dW

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  num_train = X.shape[0]

  #should work OK
  scores = X.dot(W)   # (N,D).dot(D,C) = (N examples ,C classes)  scores for each example by class

  #fanccy indexing to get correct scores (N), then reshape to get (N,1)
  correct_class_scores = scores[np.arange(len(scores)),y].reshape(len(scores),1)

  #subtract correct_class_score[i] from every element per row and add delta
  scores =scores-correct_class_scores + 1

  #How about we determine which are less than 0
  #and create an array of (N,C)  
  #if less than 0 then use 0, > 0 then 1
  vecclamp=(scores>=0).astype(int)
    
  #now lets set correct class score loc to 0
  vecclamp[np.arange(len(scores)),y] = 0

  #then mult through to clamp to 0
  scores=scores*vecclamp

  #sum over all axis on the matrix (just add every number)
  loss = np.sum(scores)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #I am going through N training examples one at a time
  #and for each class, add number of times loss >0 and put in cnt
  #multiply -cnt*xi and put in correct class from yi 
  #
  #for every other class where loss>0
  #add xi to class
  #
    
  #lets build vecclamp out
  vecclamp[np.arange(num_train),y] = -np.sum(vecclamp,axis=1)
    
  #now calculate dw
  dw=np.dot(np.transpose(vecclamp),X)
  
  #Also use the averaged gradient of full batch
  dw = np.transpose(dw)
  dw /= num_train
  
  # add regularization gradient
  dw += reg * W
  
  return loss, dw 