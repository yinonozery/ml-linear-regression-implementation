import matplotlib.pyplot as plt
from pandas import read_excel
from numpy import arange, linspace
from scipy.stats import linregress
from random import random

def scipyLinearRegression(x: list, y: list) -> tuple[int, int]:
    # Scipy Implementation
    slope, intercept, rvalue, pvalue, std_err = linregress(x, y)
    return (slope, intercept)  # (t1, t0)


def manualLinearRegression(x: list, y: list) -> tuple[list, list, float, float, list]:
    # Manual Method : Gradient Descent Implementation
    m = len(x)  # num of data examples
    alpha = 1e-4  # learning rate
    ERR = 1.0e-8  # epsilon - convergence "error" for cost function
    NumIter = 1.0e+6  # max num of iterations

    # Initilize of Thetas
    t0 = random()  # tetha0
    t1 = random()  # tetha1

    ts0 = []  # storage vector t0
    ts1 = []  # storage vector t1
    costf = []  # storage vector for cost function values

    lastCost = ERR + 1
    dcost = ERR + 1

    # Cost Function
    def cost():
        res = 0
        for k in range(0, m):
            res += ((t1 * x[k] + t0) - y[k])**2
        return res / (2*m)

    i = 0  # iteration counter

    while dcost > ERR and i < NumIter:
        i += 1

        # s0 , s1 - vector of sums of distance errors
        s0 = 0
        s1 = 0
        for j in range(0, m):
            s0 += (t0 + t1*x[j] - y[j])
            s1 += (t0 + t1*x[j] - y[j]) * x[j]

        costf.append((i, lastCost))
        # diff of Cost function between the iterations
        currCost = cost()
        dcost = abs(currCost - lastCost)

        # update lastCost
        lastCost = currCost
        # Calculate t0 , t1
        temp0 = t0 - (alpha/m) * s0
        temp1 = t1 - (alpha/m) * s1

        # Update thethas
        t0 = temp0
        t1 = temp1

        # For Control : storage vector of t0, t1
        ts0.append(t0)
        ts1.append(t1)

    return (ts0, ts1, t1, t0, costf)


def main():
    # -- Get training data -- #
    df = read_excel('lab_tests.xlsx')
    x = df['dose'][1:].to_list()
    y = df['risk'][1:].to_list()
    if len(x) != len(y):
        print("Wrong training data, x and y must be the same size")
        return

# -- Compare both implementation -- #
    t1, t0 = scipyLinearRegression(x, y)
    ts0, ts1, mt1, mt0, costf = manualLinearRegression(x, y)

    # Create multiple figures in the same window
    figure, axis = plt.subplots(
        2, 2, num='LinearRegression (Ex1)', figsize=(10, 7))
    figure.tight_layout(pad=5)


# -- Spicy line vs Our line -- #
    # Spicy line
    def spicyLinearfunc(x):
        return t1 * x + t0
    sciPy = list(map(spicyLinearfunc, x))
    axis[0][0].plot(x, sciPy, label="SciPy, \u03F4(1): {}, \u03F4(0): {}".format(
        str(t1)[:6], str(t0)[:6]))

    # Our line
    def manualLinearfunc(x):
        return mt1 * x + mt0
    manual = list(map(manualLinearfunc, x))
    axis[0][0].plot(x, manual, label="Manual, \u03F4(1): {}, \u03F4(0): {}".format(
        str(mt1)[:6], str(mt0)[:6]))

    axis[0][0].scatter(x, y)  # training data points
    axis[0][0].set_title('Dose-Response Relationship')
    axis[0][0].set_xlabel('Dose (mg)')
    axis[0][0].set_ylabel('Incidence (%)')
    axis[0][0].legend()

# -- Theta0 vs Theta1 -- #
    axis[0][1].set_title('\u03F4(0)-\u03F4(1) Relationship')
    axis[0][1].scatter(ts0, ts1)
    axis[0][1].set_xlabel('\u03F4(0)')
    axis[0][1].set_ylabel('\u03F4(1)')

# -- Iterations vs Cost Function -- #
    axis[1][0].set_title('Evolution of cost value during J minimization')
    axis[1][0].scatter(*zip(*costf))
    axis[1][0].plot()
    axis[1][0].set_xlabel('Number of Iterations')
    axis[1][0].set_ylabel('Cost Function Values')

# -- T0 (intercept) vs Cost Function -- #
    def cost(t0):
        res = 0
        for k in range(0, len(x)):
            res += ((t1 * x[k] + t0) - y[k])**2
        return res / (2*len(x))
    
    xTs0 = arange(0, t0 * 2, 0.01)
    costFunc = list(map(cost, xTs0))
    axis[1][1].set_title('T0 (intercept)-Cost Function Relationship')
    axis[1][1].scatter(t0, cost(t0), color='red')
    axis[1][1].plot(xTs0, costFunc)
    axis[1][1].set_xlabel('T0 (intercept)')
    axis[1][1].set_ylabel('Cost Function Values')

    # Plot slope 0 at minimum point
    def line(x, x1, y1):
        return 0 * (x1)*(x - x1) + y1
    xrange = linspace(t0-5, t0+5, 50)
    axis[1][1].plot(xrange, line(xrange, t0, cost(t0)), 'C1--', linewidth=2)

# -- Compare mean squared error -- #
    def costScipy():
        res = 0
        for k in range(0, len(x)):
            res += ((t1 * x[k] + t0) - y[k])**2
        return res / (2*len(x))

    def costManual():
        res = 0
        for k in range(0, len(x)):
            res += ((mt1 * x[k] + mt0) - y[k])**2
        return res / (2*len(x))

    print('Mean Squared Error:\nScipy Implementation: ', costScipy(), '\nManual Implementation:', costManual())

    # Show all graphs
    plt.show()


if __name__ == "__main__":
    main()
