import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax,cross_entropy_error
from common.gradient import numerical_gradient

class simple_Net:
    def __init__(self):
        self.W=np.random.rand(2,3) # 정규분포로 초기화

    def predict(self,x):
        return np.dot(x,self.W)

    def loss(self,x,t):
        z = self.predict(x)
        y =softmax(z)
        loss = cross_entropy_error(y,t)

        return loss

net = simple_Net()
print(net.W)

x = np.array([0.6,0.9])
p = net.predict(x)
print(p)

np.argmax(p)

t= np.array([0,0,1])
net.loss(x,t)

f = lambda w: net.loss(x,t)
dw = numerical_gradient(f,net.W)
