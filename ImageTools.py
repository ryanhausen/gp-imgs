#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np

#https://en.wikipedia.org/wiki/Image_moment

def raw_moment(img, x, y, i, j):
    return (img[y,x] * x**i * y**j).sum()

def img_center(img, src_map):
    y, x = np.where(src_map)
    moment = lambda i, j: raw_moment(img, x, y, i, j)

    m00 = moment(0, 0)
    m01 = moment(0, 1)
    m10 = moment(1, 0)

    x_centroid = int(round(m10 / m00))
    y_centroid = int(round(m01 / m00))

    return (x_centroid, y_centroid)

def img_cov(img, src_map):
    y, x = np.where(src_map)
    moment = lambda i, j: raw_moment(img, x, y, i, j)

    m00 = moment(0, 0)
    m01 = moment(0, 1)
    m10 = moment(1, 0)

    x_centroid = m10 / m00
    y_centroid = m01 / m00

    # second order central moments
    mu11 = (moment(1, 1) - x_centroid * m01) / m00
    mu20 = (moment(2, 0) - x_centroid * m10) / m00
    mu02 = (moment(0, 2) - y_centroid * m01) / m00

    return np.array([[mu20, mu11],[mu11, mu02]])

def axis_ratio(img, src_map):
    cov = img_cov(img, src_map)
    evals, _ = np.linalg.eig(cov)
    return np.sqrt(evals.min()/evals.max())

def effective_radius(img, src_map, return_ie=False):
    Itot = img[src_map].sum()
    r_vals = radial_bin_image(img, src_map)

    Re, Ie, Itmp = 0.0, 0.0, 0.0
    for r in sorted(r_vals.keys()):
        Itmp += sum(r_vals[r])
        if Itmp >= Itot/2:
            Re = r
            Ie = np.mean(r_vals[r])
            break

    return (Re, Ie) if return_ie else Re

def radial_frame(x, y, cx, cy):
    img = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            img[i,j] = np.sqrt((i-cx)**2 + (j-cy)**2) * 0.06
    return img

def fill_radial_frame(radial_frame, radial_profile):
    for x in range(radial_frame.shape[0]):
        for y in range(radial_frame.shape[1]):
            radial_frame[y,x] = radial_profile.get(radial_frame[y,x], 0.0)
    return radial_frame

def trans_to_origin():
    return np.array([
                [1.0, 0.0, 42.0],
                [0.0, 1.0, 42.0],
                [0.0, 0.0, 1.0]
            ])

def trans_from_origin():
    return np.array([
                [1.0, 0.0, -42.0],
                [0.0, 1.0, -42.0],
                [0.0, 0.0, 1.0]
            ])

def scale_image(w, h):
    return np.array([
                [w, 0.0, 0.0],
                [0.0, h, 0.0],
                [0.0, 0.0, 1.0]
            ])

def PIL_tuple(matrix):
    return tuple(matrix.flatten()[:6])

def sbp(img, src_map):
    rs = radial_bin_image(img, src_map)

    xs, ys = [], []

    for r in sorted(rs.keys()):
        xs.append(r)
        ys.append(np.mean(rs[r]))

    return xs, ys

def radial_bin_image(img,
                     src_map,
                     re_limit=4.0,
                     input_re=False,
                     input_center=False,
                     re_normed=False,
                     ie_normed=False):

    Itot = img[src_map].sum()

    cx, cy = input_center if input_center else img_center(img, src_map)

    r_vals ={}
    for y, x in zip(*np.where(src_map)):
        r = np.sqrt((x-cx)**2 + (y-cy)**2) * 0.06
        r_vals.setdefault(r, []).append(img[y,x])

    Re, Ie, Itmp, count = None, None, 0.0, 0
    found_vals = False
    if input_re==False:
        for r in sorted(r_vals.keys()):
            count += 1
            Ibin = sum(r_vals[r])
            Itmp += Ibin
            if Itmp >= Itot/2:
                found_vals = True
                Re = r
                Ie = np.mean(r_vals[Re])
                break
    else:
        Re = input_re
        Ie = np.mean(r_vals[Re])

        found_vals = True

    if found_vals == False:
        raise(Exception(f'Couldn\'t find Re. Needed to reach {Itot/2} got to {Itmp}, count={count}'))

    if ie_normed:
        for r in r_vals.keys():
            r_vals[r] = np.mean(r_vals[r])/abs(Ie)

    if re_normed:
        tmp = {}
        for r in r_vals.keys():
            if r/Re >= 5.0: continue
            tmp[r/Re] = r_vals[r]
        r_vals = tmp

    return r_vals

def signal_to_noise(img, src_map, noise_map):
    # not sure I want this in the larger namespace
    def rms(collection):
        try:
            return np.sqrt((1/len(collection))*(np.square(collection).sum()))
        except ZeroDivisionError as e:
            print(f'{e} lenth of array {len(collection)}')
            return 0.0

    re = effective_radius(img, src_map)
    cx, cy = img_center(img, src_map)
    radial_grid = radial_frame(img.shape[0], img.shape[1], cx, cy)

    s = img[radial_grid <= re].sum()

    nmap = np.logical_and(noise_map, (radial_grid >= 3*re))
    n = rms(img[nmap].flatten())

    return s/n