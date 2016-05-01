import numpy as np
X= np.array([ [0,0,1], [0,1,1], [1,0,1], [1,1,1] ])
y = np.array([[0,1,1,0]]).T

syn0 = 2*np.random.random((3,4)) -1
syn1 = 2*np.random.random((4,1)) -1
num_samples = X.shape[0]
rowskip =1000
for j in range (160000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))

    l2_delta = (y-l2)*(l2*(1-l2))   #apply chain rule to d/dl2[ (1/2)(y-l2)**2]
                                    #gives you greatest increase, wanna minimize so mult by -1
    l1_delta = l2_delta.dot(syn1.T)*(l1*(1-l1))

    #why the l1 and X?
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)

    if j%rowskip == 0:
        print("error:" + str( (np.sum(y-l2)/num_samples) ))
