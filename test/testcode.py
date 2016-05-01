import numpy as np
#W1: First layer weights; has shape (D, H)

class Car(object):
    wheels = 4

    def __init__(self, make, model):
        self.make = make
        self.model = model

    @staticmethod
    def make_car_sound():
        print ('VRooooommmm!')

tot = Car.wheels
mustang = Car('Ford', 'Mustang')
mustang.make_car_sound()
Car.make_car_sound()



###################
input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5


np.random.seed(1)
X = 10 * np.random.randn(num_inputs, input_size)  #num_inputs seperate examples, each input_size wide
y = np.array([0, 1, 2, 2, 1])                     #num_inputs correct classes

size = X.shape[0]

W1=np.random.rand(input_size,hidden_size)
b1 = np.zeros(hidden_size)

f=np.dot(X,W1)+b1

#How about we determine which are less than 0
#and create an array of (N,C)
#if less than 0 then use 0, > 0 then 1
# vecclamp=(f>=0)
# f=vecclamp*f
# f=np.max(0,f)

#########################
X= np.array([ [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1] ])
y = np.array([0,1,2,3,4,5,6,7]).T

num_train = X.shape[0]

W = np.random.randn(3,8)
scores = X.dot(W)
correct_scores = scores[range(num_train),y]


################################
textx=np.array([[-5,-4],[10,9],[-20,55],[30,6],[1,1]])
texty-=np.max(textx,axis=0) #want to find columnlar max


