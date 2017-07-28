#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:11:35 2017

@author: ryanhausen
"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt

import DataTools as dt
import ImageTools as it

files = os.listdir('.')
if ('spheroid_ratio_vals.json' not in files) or ('disk_ratio_vals.json' not in files):
    bands = ['h', 'j', 'v', 'z']
    c_band = 'h'
    count = None
    use_orig = True

    sources = []
    print('Getting Disks...')
    sources.append(('disk', dt.disks(bands=bands, count=count, log_scale=False, orig=use_orig)))
    print('Getting Spheroids...')
    sources.append(('spheroid', dt.spheroids(bands=bands, count=count, log_scale=False, orig=use_orig)))


    # returns a list of (img_id, img[bands, 84,84], segmap)
    for name, data in sources:
        vals = {b:{} for b in bands}

        for d in data:
            _id, img, seg = d
            src_map = seg == _id

            invalid = False
            for rule in [dt.all_bands_have_valid_data]:
                if rule(src_map, img) == False:
                    print(f'Img {_id} failed rule {rule}')
                    invalid = True
                    break
            if invalid: continue

            h_img = img[bands.index('h')]
            h_re = it.effective_radius(h_img, src_map)
            h_center = it.img_center(h_img, src_map)

            if h_re == 0:
                print('Re to small')
                continue


            r_bin = lambda i: it.radial_bin_image(i,
                                                  src_map,
                                                  re_limit=4.0,
                                                  input_re=h_re,
                                                  input_center=h_center,
                                                  re_normed=False,
                                                  ie_normed=False)
            b_bin = lambda b: r_bin(img[bands.index(b)])

            r_bins = {b: b_bin(b) for b in bands}

            tmp = list(r_bins[c_band].keys())
            keep = []
            for r in tmp:
                keep_r = True
                for b in bands:
                    if r not in list(r_bins[b].keys()):
                        keep_r = False
                if keep_r:
                    keep.append(r)

            for b in bands:
                for r in sorted(keep):
                    b_val = np.mean(r_bins[b][r])
                    c_val = np.mean(r_bins[c_band][r])

                    vals[b].setdefault(r, []).append(b_val / c_val)


        bin_range = (0, np.ceil(max([max(vals[b].keys()) for b in bands])))
        bins = np.linspace(bin_range[0], bin_range[1], num=max(bin_range)/0.06)

        bin_keys = {b:{} for b in bands}
        for b in bands:
            keys = list(sorted(vals[b].keys()))
            idx = np.digitize(keys, bins)
            for i in range(len(keys)):
                bin_keys[b][keys[i]] = idx[i]

        binned_vals = {b:{} for b in bands}

        percentile = lambda p, coll: np.sort(coll)[int(len(coll)*p)]

        for b in bands:
            xs, ys, std = [], [], []
            y16s, y84s = [], []

#            for r in sorted(vals[b].keys()):
#                y50 = percentile(.5, np.array(vals[b][r]).flatten())
#                y84 = percentile(.84, np.array(vals[b][r]).flatten())
#                y16 = percentile(.16, np.array(vals[b][r]).flatten())
#
#                xs.append(r)
#                ys.append(y50)
#                std.append(np.abs(y16-y50))
#                y16s.append(y16)
#                y84s.append(y84)




            for i in range(0, len(bins)):
                bin_vals = []
                ks = sorted(list(vals[b].keys()))
                for j in range(len(ks)):

                    if bin_keys[b][ks[j]] == i+1:
                        bin_vals.extend(vals[b][ks[j]])

                if len(bin_vals) > 0:
                    _vals = np.array(bin_vals).flatten()
                    _val_mask = np.abs(_vals) < 10

                    xs.append(bins[i])
                    y16 = percentile(.16, _vals[_val_mask])
                    y50 = percentile(.5, _vals[_val_mask])
                    y84 = percentile(.84, _vals[_val_mask])
                    ys.append(y50)
                    y16s.append(y16)
                    y84s.append(y84)
                    std.append(y84-y50)

            xs = np.array(xs)
            ys = np.array(ys)
            std = np.array(std)
            y16s = np.array(y16s)
            y84s = np.array(y84s)

            binned_vals[b]['x'] = xs
            binned_vals[b]['y'] = ys
            binned_vals[b]['std'] = std
            binned_vals[b]['y16'] = y16s
            binned_vals[b]['y84'] = y84s
            plt.plot(xs, ys, label=b)
            plt.fill_between(xs, y16s, y84s, alpha=0.2, label=f'$\sigma$ - {b}')
            plt.legend()

        plt.show()
        for b in bands:
            for k in binned_vals[b].keys():
                binned_vals[b][k] = dt._nmpy_encode(binned_vals[b][k])

        with open(f'{name}_ratio_vals.json', 'w') as fp:
            json.dump(binned_vals, fp)

        if False:
            plt.figure()
            for b in bands:
                c_x, c_y = binned_vals[c_band]['x'], binned_vals[c_band]['y']

                x, y = binned_vals[b]['x'], binned_vals[b]['y']

                if len(x) != len(c_x):
                    x = x if len(x) < len(c_x) else c_x
                    y = y/c_y[:len(x)] if len(y) < len(c_y) else y[:len(x)]/c_y

                plt.plot()

            plt.show()


#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF

from GaussianProcess import GP, RBF
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF


data_spheroids, data_disks = None, None

with open('spheroid_ratio_vals.json', 'r') as fs, open('disk_ratio_vals.json') as fd:
    data_spheroids, data_disks = json.load(fs), json.load(fd)

for k in data_spheroids.keys():
    for kk in data_spheroids[k].keys():
        data_spheroids[k][kk] = dt._nmpy_decode(data_spheroids[k][kk])
        data_disks[k][kk] = dt._nmpy_decode(data_disks[k][kk])

color_code = {'h':'b', 'j':'c', 'v':'m', 'z':'g' }
for name, data in [('disks', data_disks), ('spheroids', data_spheroids)]:
    plt.figure()
    plt.ylabel('$I_{band}(R)/I_h(R)$')
    plt.xlabel('$R$(arcseconds)')
    plt.title(f'{name} measured')
    for b in data.keys():
        x = data[b]['x']
        y = data[b]['y']

        y16 = np.array([max(_y, 0) for _y in  data[b]['y16']])
        #y16 = data[b]['y16']
        data[b]['std'] = np.abs(y16-y)
        y84 = data[b]['y84']
        std = np.abs(y16-y)
        plt.plot(x, y, label=b, color=color_code[b])
        #plt.fill_between(x, y-std, y+std, label=f'$\sigma$ - {b}', alpha=0.2, color=color_code[b])
        plt.fill_between(x, y-std, y+std, label=f'$\sigma$ - {b}', alpha=0.2, color=color_code[b])

    plt.legend()

plt.show()
def generate_image(Re, x, gp, idx):
    pix = 0.06
    x = x[:,0]
    print(x.shape, mu.shape)
    # get the radii values for the image
    rs = []
    for i in range(84):
        for j in range(84):
            r = (i-42)**2 + (j-42)**2
            r = np.sqrt(r)*pix
            rs.append(r)
    rs = sorted(np.unique(rs))

    # normalize the radii values to the Re param
    Res = np.array([r/Re for r in rs if r/Re <= x.max()])
    sample = gp.predict(Res[:,np.newaxis])[:,idx]

    img = np.zeros([84,84])
    for i in range(84):
        for j in range(84):
            r = (i-41)**2 + (j-41)**2
            r = np.sqrt(r)*pix

            if r/Re < x.max():
                img[i,j] = sample[Res==r/Re]

    return img

# x for the GP
# j/h, v/h, z/h are the y values
# for alpha we'll use the mean std at every point
plt.figure()
plt.ylabel('$I_{band}(r)/I_h(r)$')
plt.xlabel('$R$(arcseconds)')
plt.title(f'{name} GP')
x = data['h']['x']
y = data['j']['y']
al = data['j']['std']
plt.plot(x, y)
plt.fill_between(x, y-al, y+al, alpha=0.2)


x = data['h']['x'][:, np.newaxis]
y = data['j']['y'][:, np.newaxis]
al = data['j']['std']


p = RBF.init_params()
p[0] = 1.0 # length-scale
p[1] = 1.0 # f_n
kernel = RBF(p)
gp = GP(kernel)
gp.fit(x, y, al, optimize=True)
mu, std = gp.predict(x)
std = np.diag(std)
plt.plot(x[:,0], mu)
mu = mu[:,0]
x = x[:,0]

plt.fill_between(x, mu-std, mu+std, alpha=0.2)
plt.show()



not_h = ['j', 'v', 'z']
for name, data in [('Disks', data_disks), ('Spheroids', data_spheroids)]:
    plt.figure()
    plt.ylabel('$I_{band}(r)/I_h(r)$')
    plt.xlabel('$R$(arcseconds)')
    plt.title(f'{name} GP')
    x = data['h']['x']
    x = x.reshape(x.shape[0], 1)

    y = tuple([data[b]['y'] for b in not_h])
    y = np.stack(y, axis=1)

    alpha = [data[b]['std'] for b in not_h]
    alpha = np.stack(alpha, axis=1).mean(axis=1)

    p = RBF.init_params()
    p[0] = .1 # length-scale
    p[1] = 1000 # f_n
    kernel = RBF(p)
    gp = GP(kernel)
    gp.fit(x, y, alpha, optimize=False)
    gp.save(f'{name.lower()[:-1]}_gp_colors.json')

    print(-gp.loglikelihood())


    mu, std = gp.predict(x)
    std = np.diag(std)

    for i in range(mu.shape[1]):
         plt.plot(x, mu[:,i], label=not_h[i], color=color_code[not_h[i]])
         plt.fill_between(x[:,0], mu[:,i]-std, mu[:,i]+std, color=color_code[not_h[i]], alpha=0.2)
    plt.ylim((-2, 5))
    plt.legend()


plt.show()


















