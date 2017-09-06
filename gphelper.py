import json

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Product

from scipy.optimize import fmin

import DataTools as dt

class GPHelper(object):
    def __init__(self,restore_file=None):
        if restore_file:
            X, Y, a, k = GPHelper._load_params(restore_file)
            self.X = X
            gp = GaussianProcessRegressor(kernel=k, alpha=a, optimizer=None)
            self._gp = gp.fit(X, Y)


    @staticmethod
    def _load_params(rf):
        with open(rf, 'r') as f:
            params = json.load(f)

        X = dt._nmpy_decode(params['X'])
        Y = dt._nmpy_decode(params['Y'])
        alpha = dt._nmpy_decode(params['alpha'])
        k = eval(params['k'])

        return X, Y, alpha, k

    def save_params(self, save_file=None):
        # we need to save X, Y, alpha, and k to restore this same GP
        params = {
            'X':dt._nmpy_encode(self._gp.X_train_),
            'Y':dt._nmpy_encode(self._gp.y_train_),
            'alpha':dt._nmpy_encode(self._gp.alpha),
            'k':str(self._gp.kernel_)
        }

        save_file = save_file if save_file else 'gp.json'
        with open(save_file, 'w') as f:
            json.dump(params, f, sort_keys=True, indent=4)

    def fit(self, X, Y, alpha, length_scale=1.0, sigma_n=1.0, optimize=None):
        self.X = X
        kernel = Product(ConstantKernel(length_scale), RBF(length_scale=length_scale))
        make_gp = lambda k, o: GaussianProcessRegressor(kernel=k, alpha=alpha, optimizer=o)
        gp = make_gp(kernel, None).fit(X, Y)

        def optimize_sigma_n(n):
            theta = (np.log(n), np.log(length_scale))
            return -gp.log_marginal_likelihood(theta=theta)

        def optimize_length_scale(l):
            theta = (np.log(sigma_n), np.log(l))
            return -gp.log_marginal_likelihood(theta=theta)



        if optimize=='sigma_n':
            sigma_n = fmin(optimize_sigma_n, [sigma_n])[0]
        elif optimize=='length_scale':
            length_scale = fmin(optimize_length_scale, [length_scale])[0]

        kernel = Product(ConstantKernel(sigma_n), RBF(length_scale=length_scale))
        self._gp = make_gp(kernel, None).fit(X, Y)

        if optimize=='both':
            self._gp = make_gp(kernel,'fmin_l_bfgs_b').fit(X, Y)

        return self

    def predict(self, X, return_std=False, return_cov=False):
        return self._gp.predict(X, return_std=return_std, return_cov=return_cov)

    def sample(self, X, num_samples=1, monotonic=True, seed=None):
        # we need to ensure that all samples are monotonically decreasing
        mono_dec = lambda s: np.all(np.diff(s) <= 0)

        samples = []
        for i in range(num_samples):
            count = 1

            seed = seed if seed else np.random.randint(0, 1e9)
            sample = self._gp.sample_y(X, random_state=np.random.randint(0, 1e9))
            while (mono_dec(sample.flatten())==False and monotonic):
                num_dots = count % 4
                count += 1
                print('Drawing Samples'+'.'*num_dots+' '*(5-num_dots), end='\r')
                seed = np.random.randint(0, 1e9)
                sample = self._gp.sample_y(X, random_state=np.random.randint(0, 1e9))
            samples.append(sample.flatten()[:,np.newaxis])

        if len(samples)==1:
            samples=samples[0]
        samples = np.atleast_2d(np.array(samples))

        return samples
