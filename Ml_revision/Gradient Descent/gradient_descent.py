import pandas as pd

data = pd.read_csv('data.csv')

x = data.iloc[:, 1].values
y = data.iloc[:, -1].values

print("x: ", x)
print("y", y)


def gradient_descent(x, y):
    m = c = 0
    iterations = 100000
    n = len(x)
    learning_rate = 0.0001
    for i in range(iterations):
        y_predicted = m * x + c
        cost = (1 / n) * sum([value ** 2 for value in (y - y_predicted)])
        # calculate derivative
        md = -(2 / n) * sum(x * (y - y_predicted))
        cd = -(2 / n) * sum(y - y_predicted)

        # calculate m and c values
        m = m - learning_rate * md
        c = c - learning_rate * cd

        # printing each iteration
        print(f'm:{m}, c:{c}, cost:{cost}, iteration:{i}')

gradient_descent(x,y)