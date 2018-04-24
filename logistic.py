import numpy as np
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_train = train_set_x_orig.shape[0]  #训练样本数目
m_test = test_set_x_orig.shape[0]   #测试样本数目
num_px = train_set_x_orig.shape[1]  #图像的高度

train_set_x_flatten = train_set_x_orig.reshape(m_train,-1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test,-1).T

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255


def sigmoid(z):
    return 1/(1+np.exp(-z))

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    assert(w.shape == (dim,1))
    return w,b

def propagate(w,b,X,Y):
    
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = (-1)* np.sum(np.dot(Y,np.log(A).T)+np.dot((1-Y),np.log(1-A).T))/m
    dw = np.dot(X,(A-Y).T) / m
    db = np.sum(A-Y) / m

    cost = np.squeeze(cost)

    grads = {"dw" : dw, "db" : db}
    return grads,cost

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w,b,X,Y)

        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate*dw
        b = b - learning_rate * db

        if i%100 == 0:
            costs.append(cost)
        if print_cost and i%100 == 0:
            print("Cost after iteration %i :%f" % (i,cost))
        
    params = {"w" : w, "b" :b}
    grads = {"dw" : dw , "db" : db}
    return params, grads, costs

def predict(w,b,X):

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            Y_prediction[ 0,i] = 1
        else:
            Y_prediction[ 0 ,i] = 0
    return Y_prediction


def model(X_train , Y_train, X_test, Y_test, num_iterations = 2000,learning_rate = 0.5,print_cost=False):
    w , b = initialize_with_zeros(64*64*3)

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations,learning_rate,print_cost=False)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
    "Y_prediction_test": Y_prediction_test,
    "Y_prediction_train": Y_prediction_train,
    "w": w,
    "b": b,
    "learning_rate": learning_rate,
    "num_iterations": num_iterations}

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000,learning_rate=0.005,print_cost=True)

