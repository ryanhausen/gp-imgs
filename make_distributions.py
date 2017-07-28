#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:12:06 2017

@author: ryanhausen

"""
import json

import numpy as np
import matplotlib.pyplot as plt

import DataTools as dt

def main():
    print('Getting spheroids...')
    s_ar, s_re = dt.spheroid_ar_re('h')
    print('Getting disks...')
    d_ar, d_re = dt.disk_ar_re('h')

    rec_f = lambda f, s, d : f([f(s), f(d)])

    ar_rng = (rec_f(min, s_ar, d_ar), rec_f(max, s_ar, d_ar))
    re_rng = (rec_f(min, s_re, d_re), rec_f(max, s_re, d_re))
    bins = int(np.ceil((re_rng[1] - re_rng[0]) / 0.06) * 2)
    print(bins)

    print('Making graphs...')
    ar_re_gauss = dict()
    for name, data in [('Disk', (d_ar, d_re)), ('Spheroid', (s_ar, s_re))]:
        axis_ratio, re = data
        
        
        print(name)
        print(f'Unique Values AR:{len(np.unique(axis_ratio))}/{len(axis_ratio)}')
        print(f'Unique Values RE:{len(np.unique(re))}/{len(re)}')

        min_amt = 5
        
        ar_vals, ar_edges = np.histogram(axis_ratio, bins=bins, range=ar_rng)
        ar_vals[ar_vals<min_amt] = 0
        re_vals, re_edges = np.histogram(re, bins=bins, range=re_rng)
        re_vals[re_vals<min_amt] = 0

        data = {
            'ar_vals':dt._nmpy_encode(ar_vals),
            'ar_bins':dt._nmpy_encode(ar_edges),
            're_vals':dt._nmpy_encode(re_vals),
            're_bins':dt._nmpy_encode(re_edges)
        }

        with open(f'{name.lower()}_bins.json', 'w') as fp:
            json.dump(data, fp)

        f, a = plt.subplots(2,1)
        f.suptitle(name)

        a[0].set_title('Axis Ratio')
        a[0].hist(axis_ratio, bins=bins, range=ar_rng)

        a[1].set_title('Effective Radius')
        a[1].hist(re, bins=bins, range=re_rng)

        f.tight_layout()
        
        plt.figure()
        plt.title(f'{name} Ratio Scatter Plot')
        plt.ylabel('Re')
        plt.xlabel('Axis Ratio')
        plt.xlim((0,1.5))
        plt.ylim((0,1.5))
        plt.scatter(axis_ratio, re, color='b')
        
        # fit a guassian
        plt.figure()
        plt.title(f'{name} 100 Samples')
        plt.ylabel('Re')
        plt.xlabel('Axis Ratio')
        plt.xlim((0,1.5))
        plt.ylim((0,1.5))
        d = np.array([axis_ratio, re])
        cov = np.cov(d)
        mu = np.array(np.mean(d, axis=1))
        xmin, xmax = axis_ratio.min(), axis_ratio.max()
        ymin, ymax = re.min(), re.max()
        
        samples = np.random.multivariate_normal(mu, cov, 100)
        samples = []
        while(len(samples) < 100):
            sample = np.random.multivariate_normal(mu, cov, 1)
            _ar = sample[0, 0]
            _re = sample[0, 1]
            if (_ar > xmin and _ar < xmax) and (_re > ymin and _re < ymax):
                samples.append(sample[0,:])
        
        samples = np.array(samples)
        print(samples.shape)
        
        plt.scatter(axis_ratio, re, color='b', label='Data')
        plt.scatter(samples.T[0,:], samples.T[1,:], color='r', label='Samples')
        
        plt.axhline(ymin, xmin=xmin/1.5, xmax=xmax/1.5, color='g')
        plt.axhline(ymax, xmin=xmin/1.5, xmax=xmax/1.5, color='g')
        
        plt.axvline(xmin, ymin=ymin/1.5, ymax=ymax/1.5, color='g')
        plt.axvline(xmax, ymin=ymin/1.5, ymax=ymax/1.5, color='g', label='Rejection Criteria')
        
        plt.legend()
        
        
        
        
        delta = 0.025
        _x = np.arange(0, 1.5, delta)
        _y = np.arange(0, 1.5, delta)
        X,Y = np.meshgrid(_x, _y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        
        ar_re_gauss[name.lower()] = dict()
        ar_re_gauss[name.lower()]['mu'] = dt._nmpy_encode(mu)
        ar_re_gauss[name.lower()]['cov'] = dt._nmpy_encode(cov)
        
        from scipy.stats import multivariate_normal
        rv = multivariate_normal(mu, cov)
        
        plt.figure()
        plt.title(f'{name} 2D Gaussian')
        plt.ylabel('Re')
        plt.xlabel('Axis Ratio')
        plt.xlim((0,1.5))
        plt.ylim((0,1.5))
        #plt.scatter(axis_ratio, re, color='b')
        plt.contour(X, Y, rv.pdf(pos))
        
        # Recreating the distribution
        samples = []
        while(len(samples) <  len(axis_ratio)):
            sample = np.random.multivariate_normal(mu, cov, 1)
            _ar = sample[0, 0]
            _re = sample[0, 1]
            if (_ar > xmin and _ar < xmax) and (_re > ymin and _re < ymax):
                samples.append(sample[0,:])
        
        samples = np.array(samples)
        print(samples.shape)
        
        f, a = plt.subplots(2,1)
        f.suptitle(f'{name} Recovering Historgram')

        a[0].set_title('Axis Ratio')
        a[0].hist(axis_ratio, bins=bins, range=ar_rng, color='b', label='data')
        a[0].hist(samples[:, 0], bins=bins, range=ar_rng, color='r', label='samples')

        a[1].set_title('Effective Radius')
        a[1].hist(re, bins=bins, range=re_rng, color='b', label='data')
        a[1].hist(samples[:,1], bins=bins, range=re_rng, color='r', label='samples')

        f.tight_layout()
        
        
        
    with open('ar_re_gaussian.json', 'w') as f:
        json.dump(ar_re_gauss, f)
        

    plt.show()


    




if __name__=='__main__':
    main()
