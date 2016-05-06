#!/usr/bin/env python
# ex: set expandtab ts=4 sts=4 sw=4:

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys


def linear_regression(x, y, theta, alpha, num_iter):
    '''x, the input matrix is a m x (n+1) 2d-array where n is the number of properties and m is number of records.
    y, the output matrix is a m x 1 2d-array.
    theta is (n+1) x 1 2d-array.
    alpha, the learning rate is a float.
    num_iter is an int. It's the number of iteration.'''
    
    h = lambda x: x.dot(theta)
    m = y.shape[0]
    J = []
    for i in xrange(num_iter):
        loss = h(x) - y
        gradient = x.T.dot(loss) / m   #(1/m) * sigma[i= 1 to m] (h(x(i))-y(i))*x(i,j)
        j = 1/(2.0*m) * (loss * loss).sum()
        #print >>sys.stderr, '{i}# j = {j}'.format(i=i, j=j)
        theta -= alpha * gradient
        J.append(j)
    return J

def logistic_regression(x, y, theta, alpha, num_iter):
    '''x is (m x n+1) 2D array - the input data
    y is (m x 1) array - the output corresponding to x
    thets is (n+1 x 1) 2D array - the parameters
    alpha is a scalar - the scaling factor
    num_iter is a scalar - number of times the algo iterates while learning
    '''
    h = lambda x: 1.0 / (1 + np.exp(-x.dot(theta)))
    m = y.shape[0]
    J = []
    for i in xrange(num_iter):
        loss = h(x) - y
        gradient = x.T.dot(loss) / m    # (1/m) * sigma[i=1 to m] (h(x(i)) - y(i)) * x(i,j)
        theta -= alpha * gradient
        j = -1.0/m * ( y * np.log(h(x)) + (1-y) * np.log(1 - h(x)) ).sum()     # -1/m * sigma[i=1 to m] ( y[i]*log( h(x[i]) ) - (1-y[i])*( 1 - log( h(x[i]) ) ) )
        J.append(j)
    return J

def read_csv(filename, normalize=False, do_normalization=None):
    with open(filename) as fd:
        data = list(csv.reader(fd))
    matrix = np.array(data, dtype=float)
    x, y = np.split(matrix, [matrix.shape[1]-1], axis=1)
    if normalize:
        if do_normalization is None:
            min_x = x.min(axis=0)
            max_x = x.max(axis=0)
            avg_x = x.sum(axis=0) / x.shape[0]
            do_normalization = lambda x: (x - avg_x) / (max_x - min_x + 1)
        x = do_normalization(x)
    x = np.hstack([np.ones(shape=(x.shape[0],1)), x])
    return (x, y, do_normalization)

def main():
    mat_x, mat_y, normalize = read_csv('datasets/logistic-regression/aps/aps1-train.csv', normalize=True)
    theta = np.ones((mat_x.shape[1],1))
    print 'Training...\n'
    #J = linear_regression(mat_x, mat_y, theta, 0.1, 3000)
    J = logistic_regression(mat_x, mat_y, theta, 0.1, 3000)
    print 'Final J = {0}'.format(J[-1])
    print '*** theta =', theta.T.tolist()

    test_x, test_y, normalize = read_csv('datasets/logistic-regression/aps/aps1-test.csv', normalize=True, do_normalization=normalize)
    #result_y = test_x.dot(theta)
    result_y = 1.0/ ( 1 + np.exp (-test_x.dot(theta)) )
    print '-'*80
    print 'Expected\t\t\tResult'
    correct_predictions = 0
    for expected, result in zip(test_y, result_y):
        print '{0}\t\t\t{1}'.format(expected, int(result+0.5))
        correct_predictions += 1 if int(expected) == int(result+0.5) else 0
    print '-'*80
    print 'Accuracy =', float(correct_predictions)/len(test_y) * 100, '%\t', correct_predictions, 'out of', len(test_y)
    print '-'*80

    plt.plot(J)
    plt.xlabel('Iteration -->')
    plt.ylabel('Cost (J) -->')
    plt.show()

if __name__ == '__main__':
    main()

