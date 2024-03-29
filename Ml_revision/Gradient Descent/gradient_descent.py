'''
    Gradient descent:
        - it is optimization algorithm
        - gradient descent used for find best fit line for training dataset

    important terms:
        - cost function>> its is used to check performance of model
        - learning rate >> the rate of which model learn from data
'''
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

x = data.iloc[:, 1].values
y = data.iloc[:, -1].values

print("x: ", x)
print("y", y)


def gradient_descent(x, y):
    m = c = 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.0001
    l1 = []
    for i in range(iterations):
        y_predicted = m * x + c  # linear regression
        cost = (1 / n) * sum([value ** 2 for value in (y - y_predicted)])  # here i use mse
        # calculate derivative
        md = -(2 / n) * sum(x * (y - y_predicted))
        cd = -(2 / n) * sum(y - y_predicted)

        # calculate m and c values
        m = m - learning_rate * md
        c = c - learning_rate * cd

        # printing each iteration
        print(f'm:{m}, c:{c}, cost:{cost}, iteration:{i}')
        l1.append(cost)


# plot graph for gradient descent
#     print("list: ", l1)
'''    plt.plot(range(iterations), l1)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Gradient Descent Convergence')
    plt.show()'''


gradient_descent(x, y)
