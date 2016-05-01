#!/usr/bin/env python
''' simple_backprop_softmax.py backprop demo with softmax
binary to decimal neural net decoder
takes 3 binary digits and converts to decimal equiv using an 8 output softmax
'''
__author__   = "Keith Perkins"
__credits__  = "basic-python-network  on iamtrask"


import numpy as np
import cs231n.classifiers.softmax as sm

X= np.array([ [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1] ])
y = np.array([0,1,2,3,4,5,6,7])


W0 = 2*np.random.random((3,4)) -1
W1 = 2*np.random.random((4,8)) -1
num_train       = X.shape[0]
rowskip         = 1000
numbiteriter    = 100000

#training
for j in range (numbiteriter):

    l1 = 1/(1+np.exp(-(np.dot(X,W0))))   #sigmoid function

    norm_scores = sm.softmax_get_normalized_scores(l1,W1)
    # if j==numbiteriter-1:
    #     print(str(norm_scores))
    loss = sm.softmax_get_data_loss(norm_scores, num_train, y)

    norm_scores_delta =  -sm.get_dScores(norm_scores, num_train,y)

    l1_delta = norm_scores_delta.dot(W1.T)*(l1*(1-l1))

    #why the l1 and X?
    W1 += l1.T.dot(norm_scores_delta)
    W0 += X.T.dot(l1_delta)

    if j%rowskip == 0:
        print("error:" + str(loss))


input=np.array([0,0,7])


