from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import random


def main():
    p = 1.5
    q = -10

    x = []
    y = []

    for a in range(0, 100, 5):
        x.append(a)

    for a in range(0, 100, 5):
        # y = (mx + c) + random value
        # The predicted line should have m close to p, c close to q
        y.append(p*a + q + float(random.randint(-100, 100)/10))

    # --------------------------------------------------------------------------
    # Predicted Line
    # y = mx + c
    m = 0
    c = 0

    learning_rate = 0.0000005
    epochs = 1000
    loss_value = epochs
    prev_loss = 0
    loss_diff = 1

    loss = []
    n_epoch = []

    # for e in range(epochs):
    e = 0
    while(loss_diff > 0.00001):
        dm = 0
        dc = 0
        sum1 = 0
        sum2 = 0
        err = 0
        prev_loss = loss_value
        e = e+1

        for k in range(len(x)):
            sum1 = sum1 + (x[k]*(y[k]-(m*x[k]+c)))
            sum2 = sum2 + (y[k]-(m*x[k]+c))
            err = err + ((y[k]-(m*x[k]+c))*(y[k]-(m*x[k]+c)))

        dm = (-2/len(x))*sum1
        dc = (-2/len(x))*sum2
        loss_value = err/len(x)
        m = m-learning_rate*dm
        c = c-learning_rate*dc

        n_epoch.append(e+1)
        loss.append(loss_value)
        loss_value = sqrt(loss_value)

        loss_diff = prev_loss - loss_value
        print(f"Epoch: {e}, loss: {loss_value}")

    print(f"m= {m}, c= {c}")

    y_preds = []
    for a in range(len(x)):
        y_preds.append(float(m*x[a] + c))

    # print(len(x), len(y_preds))
    plt.figure(1)
    plt.plot(n_epoch, loss, c="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)

    plt.figure(2)
    plt.plot(x, y_preds, c="red")
    plt.scatter(x, y, c="blue")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    plt.plot(x, y)

    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Linear Regression")
    plt.grid(True)
    plt.show()
