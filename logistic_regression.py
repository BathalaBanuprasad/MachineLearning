import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## Generate random data
def generate_data():
    m = 3
    b = -2
    x = 5*np.random.randn(1000,)
    y_data = m*x+b
    y = np.zeros(x.shape, dtype=np.int64)
    y[y_data>=0] = 1
    y[y_data<0] = 0
    # add noise
    x+=0.1*np.random.randn(1000,)
    return x,y

def cross_entropy_loss(y, y_hat):
    return np.mean(-y*np.log(y_hat)-(1-y)*np.log(1-y_hat))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic_regression(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    w = np.random.randn()
    b = np.random.randn()
    lr = 1e-1
    loss = []
    loss_x = []
    for i in range(100):
        # fwd pass
        y_hat = w*x_train+b
        # gradient computation
        dz = y_hat-y_train
        dw = np.mean(x_train*dz)
        db = np.mean(dz)

        # update parameters
        w = w-lr*dw
        b = b-lr*db

        # test loss
        loss.append(cross_entropy_loss(y_test, w*x_test+b))
        loss_x.append(i)

    plt.figure()
    plt.plot(loss_x,loss, label="Test Loss")
    plt.title("Test Loss")
    plt.legend()

    plt.figure()
    plt.scatter(x,y, label="Data points")
    x_plot=np.linspace(-3,3)
    y_plot=w*x_plot+b
    plt.plot(x_plot,y_plot, label="Decision boundary")
    plt.plot(x_plot,sigmoid(y_plot), label="Decision boundary")
    plt.ylim([0,1])
    plt.legend()



if __name__=="__main__":
    x,y = generate_data()
    plt.scatter(x,y, label="data points")
    plt.title("Data points")
    logistic_regression(x, y)