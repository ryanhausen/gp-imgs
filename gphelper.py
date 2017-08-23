import json

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

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
        
    def fit(self, X, Y, alpha):
        self.X = X
        self._gp = GaussianProcessRegressor(alpha=alpha).fit(X, Y)
        return self
        
    def predict(self, X, return_std=False, return_cov=False):
        return self._gp.predict(X, return_std=return_std, return_cov=return_cov)
        
    def sample(self, X, num_samples=1, monotonic=True, smooth=False):
        # we need to ensure that all samples are monotonically decreasing
        mono_dec = lambda s: np.all(np.diff(s) <= 0)
        
        samples = []
        for i in range(num_samples):
            count = 1
            
            sample = self._gp.sample_y(X, random_state=np.random.randint(0,1e9))
            while mono_dec(sample.flatten())==False and monotonic:
                num_dots = count % 4
                count += 1
                print('Drawing Samples'+'.'*num_dots+' '*(5-num_dots), end='\r')
                sample = self._gp.sample_y(X, random_state=np.random.randint(0,1e9))
                
            if smooth:
                sample = dt.loessc(X,sample,1.5)
            samples.append(sample.flatten()[:,np.newaxis])
            
        if len(samples)==1:
            samples=samples[0]
        samples = np.atleast_2d(np.array(samples))
        
        return samples
