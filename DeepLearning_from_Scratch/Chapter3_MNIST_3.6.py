import sys, os
import numpy as np
import pickle
sys.path.append(os.pardir) # 부모 디렉터리 파일 가져올 수 있도록 설정
from dataset.mnist import load_mnist
from NN_chapter3 import sigmoid,softmax,relu
def get_data():
    (x_train , y_train) , (x_test, y_test) = load_mnist(flatten=True,normalize=True,one_hot_label=False)
    return x_test, y_test

def init_network():
    with open("C:/Users/Ryu/PycharmProjects/PPP/ch03/sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
    return network

def predict(network,x,Active='sigmoid'):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    if Active == 'relu':
        a1 = np.dot(x, W1) + b1
        z1 = relu(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = relu(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)
        return y
    else:
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)
        return y

x_test, y_test = get_data()
network = init_network()

accuracy_cnt =0
for i in range(len(x_test)):
    y = predict(network,x_test[i],Active='sigmoid')
    print(y)
    p = np.argmax(y)
    if p == y_test[i]:
        accuracy_cnt +=1

print("Accuracy: "+str(float(accuracy_cnt/len(x_test))))
