from __future__ import division, print_function
from abc import ABCMeta, abstractmethod

import numpy as np
import json

from scipy.optimize import fmin
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

import DataTools as dt


class GP(object):
    def __init__(self, kernel):
        self.kernel = kernel

    def _fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.K = self.kernel.k(X, X)
        self.L = np.linalg.cholesky(self.K + self.sigma_n)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y))

    def save(self, filename='gp_model.json'):

        model = dict()
        model['X'] = dt._nmpy_encode(self.X)
        model['Y'] = dt._nmpy_encode(self.Y)
        model['K'] = dt._nmpy_encode(self.K)
        model['L'] = dt._nmpy_encode(self.L)
        model['a'] = dt._nmpy_encode(self.alpha)
        model['sigma_n'] = dt._nmpy_encode(self.sigma_n)

        with open(filename, 'w') as f:
            json.dump(model, f)

    def restore(self, filename='gp_model.json'):
        with open(filename, 'r') as f:
            model = json.load(f)

        self.X = dt._nmpy_decode(model['X'])
        self.Y = dt._nmpy_decode(model['Y'])
        self.K = dt._nmpy_decode(model['K'])
        self.L = dt._nmpy_decode(model['L'])
        self.alpha = dt._nmpy_decode(model['a'])
        self.sigma_n = dt._nmpy_decode(model['sigma_n'])

    def fit(self, X, Y, sigma_n, optimize=True):
        self.sigma_n = np.diag(sigma_n**2)

        if optimize:
            def opt_fit(_params):
                self.kernel.set_params(_params)
                self._fit(X, Y)
                return -self.loglikelihood()

            p = self.kernel.init_params()
            opt_p = fmin(opt_fit, p, maxiter=10e100)
            print(f'Optimized Result: {opt_p}')

            self.kernel.set_params(opt_p)
            self._fit(X, Y)
        else:
            self._fit(X, Y)

    def predict(self, X_):
        K_s = self.kernel.k(self.X, X_)
        K_ss = self.kernel.k(X_, X_)

        mu = K_s.T.dot(self.alpha)
        v = np.linalg.solve(self.L, K_s)
        cov = K_ss - v.T.dot(v)

        self.mu = mu
        self.cov = cov

        return mu, cov

    def loglikelihood(self):
        val = -0.5 * self.Y.T.dot(self.alpha) - np.diag(self.L).sum() - (np.diag(self.sigma_n).shape[0]/2)*np.log(2*np.pi)
        return val.mean()

    def sample(self, X_, num_samples=1, return_var=False, monotonic=True, random_state=None):
        mu, cov = self.predict(X_)

        normal = random_state if random_state else np.random

        samples = None
        if monotonic == False:
            samples = normal.multivariate_normal(mu[:,0], cov, num_samples)
        else:
            mono = lambda a: np.all(np.diff(a) <= 0)

            samples = []
            for i in range(num_samples):
                sample = normal.multivariate_normal(mu[:,0], cov)

                count = 1
                while mono(sample) == False:
                    num_dots = count % 4
                    count += 1
                    print('Drawing Samples'+'.'*num_dots+' '*(5-num_dots), end='\r')
                    sample = normal.multivariate_normal(mu[:,0], cov)

                samples.append(sample)

            samples = np.atleast_2d(np.array(sample))


        return (samples, np.diag(cov)) if return_var else samples

class Kernel(metaclass=ABCMeta):
    @abstractmethod
    def init_params():
        pass

    @abstractmethod
    def k(self, x, x_):
        pass

class RBF(Kernel):
    def __init__(self, params=[1.0, 1.0]):
        self.l = params[0]
        self.f = params[1]

    @staticmethod
    def init_params():
        return [1.0, 1.0]

    def set_params(self, params):
        self.l = params[0]
        self.f = params[1]

    def k(self, x, x_):
        return self.f**2 * np.exp(-.5 * cdist(x, x_, metric='sqeuclidean')/self.l**2)

class RQ(Kernel):
    def __init__(self, params):
        self.l = params[0]
        self.f = params[1]
        self.a = params[2]

    @staticmethod
    def init_params():
        return [1.0, 1.0, 1.0]

    def k(self, x, x_):
        return self.f**2 * (1 + cdist(x, x_, metric='sqeuclidean') / (2*self.a*self.l**2))**-self.a


def main():
    # FLAGS -------------------------------------------------------------------
    PAD_VALUES = True
    OPT_NEG_LOGLIKELIHOOD = True
    # FLAGS -------------------------------------------------------------------

    # get data
    x, y16, y50, y84 = dt.spheroid_sbp('h', log_scale=True)

    if PAD_VALUES:
        pre_eval_points = 10
        pre_num_points = 5

        post_eval_points = 3
        post_num_points = 10

        # get xs
        diff = np.diff(x).mean()
        pre_x = np.linspace(x.min()-pre_num_points*diff, x.min()-diff, pre_num_points)
        post_x = np.linspace(x.max()+diff , x.max()+post_num_points*diff, post_num_points)

        pre_y = lambda y : np.poly1d(np.polyfit(x[:pre_eval_points], y[:pre_eval_points], 2))(pre_x)

        post_y = None
        if post_eval_points > 0:
            post_y = lambda y : np.poly1d(np.polyfit(x[-post_eval_points:], y[-post_eval_points:], 1))(post_x)
        else:
            post_y = lambda y : np.array([])

        extend_y = lambda y : np.concatenate((pre_y(y), y, post_y(y)))

        tmp_x = np.concatenate((pre_x, x, post_x))
        tmp_y16 = extend_y(y16)
        tmp_y50 = extend_y(y50)
        tmp_y84 = extend_y(y84)

        x, y16, y50, y84 = tmp_x, tmp_y16, tmp_y50, tmp_y84

    X = x.reshape(x.shape[0],1)
    Y = y50.reshape(y50.shape[0], 1)
    sigma_n = y84 - y50
    k = RBF(RBF.init_params())

    gp = GP(k)
    gp.fit(X, Y, sigma_n)
    gp.save('spheroid_gp_model.json')

    X_ = X.copy()

    mu, cov = gp.predict(X_)
    std = np.sqrt(np.diag(cov))

    plt.xlim((0,5))

    plt.plot(X_, mu, color='r')
    plt.fill_between(X_[:,0], mu[:,0]-std, mu[:,0]+std, alpha=0.2, color='r')

    plt.plot(x, y50, color='b')
    plt.fill_between(x, y16, y84, color='b', alpha=0.2)

    plt.show()

if __name__=='__main__':
    main()