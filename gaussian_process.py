from __future__ import division, print_function
from collections import namedtuple

import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import DataTools as dt
import ImageTools as it

def generate_image(Re, OverRe, gp, return_sample=False):
    pix = 0.06

    # get the radii values for the image
    rs = []
    for i in range(84):
        for j in range(84):
            r = (i-42)**2 + (j-42)**2
            r = np.sqrt(r)*pix
            rs.append(r)
    rs = sorted(np.unique(rs))

    # normalize the radii values to the Re param
    Res = np.array([r/Re for r in rs if r/Re <= OverRe])
    sample = gp.sample_y(Res[:,np.newaxis], random_state=np.random.randint(1000))[:,0]

    img = np.zeros([84,84])
    for i in range(84):
        for j in range(84):
            r = (i-41)**2 + (j-41)**2
            r = np.sqrt(r)*pix

            if r/Re <= OverRe:
                img[i,j] = sample[Res==r/Re]

    return (img, (Res,sample)) if return_sample else img

Plot = namedtuple('Plot', ['func', 'args', 'kwargs'])

def default_graph(plots, title):
    plt.figure()
    plt.title(title)
    plt.ylabel('$I/I_e$ Log-Scale')
    plt.xlabel('$R/R_e$')

    for p in plots:
        p.func(*p.args, **p.kwargs)

    plt.legend()

def _translate(u,v):
    return np.array([
            [1.0, 0.0, float(u)],
            [0.0, 1.0, float(v)],
            [0.0, 0.0, 1.0]
        ])

def _scale(a,b):
    return np.array([
            [a, 0.0, 0.0],
            [0.0, b, 0.0],
            [0.0, 0.0, 1.0]
        ])


# FLAGS-------------------------------------------------------------------------
GRAPH_MOCK = False
GRAPH_MEASURED = False
GRAPH_PRIOR = False
GRAPH_POSTERIOR = False
GRAPH_RECOVERED_MEAN = False
GENERATE_IMAGE = True
GRAPH_SAMPLE_STD = False
GRAPH_SAMPLES = None # replace with number of desired samples


# FLAGS-------------------------------------------------------------------------
x,y16,y50,y84 = dt.disk_sbp('h', log_scale=False)

orig_length = x.shape[0]

x = x[:, np.newaxis]
y16 = np.log(y16)[:, np.newaxis]
y50 = np.log(y50)[:, np.newaxis]
y84 = np.log(y84)[:, np.newaxis]
std = (y84-y50).flatten()

# add negative mock points
mock_fit = np.polyfit(x[:,0], y50[:,0], 2)
mock_fit = np.poly1d(mock_fit)
diff = x[1,0] - x[0,0]
mock_num = 4
mock_x = np.linspace((-mock_num * diff), (x[0,0]-diff), mock_num)
mock_50 = mock_fit(mock_x)

mock_fit = np.polyfit(x[:3,0], y84[:3,0], 1)
mock_fit = np.poly1d(mock_fit)
mock_84 = mock_fit(mock_x)
mock_std = mock_84-mock_50


if GRAPH_MOCK:
    plots = [
        Plot(func=plt.plot, args=[x,y50,'.'], kwargs={'color':'b','label':'measured'}),
        Plot(func=plt.fill_between, args=[x[:,0],y16[:,0],y84[:,0]], kwargs={'alpha':0.2, 'color':'b','label':'$\pm\sigma$'}),
        Plot(func=plt.plot, args=[mock_x,mock_50,'.'], kwargs={'color':'r','label':'Mocked'}),
        Plot(func=plt.fill_between, args=[mock_x,mock_50-mock_std,mock_50+mock_std], kwargs={'alpha':0.2,'color':'r','label':'$\pm\sigma$'})
    ]

    default_graph(plots, "Measured data With Mock Data")


if GRAPH_MEASURED:
    plots = [
        Plot(func=plt.plot, args=[x,y50,'.'], kwargs={'color':'b','label':'Median'}),
        Plot(func=plt.fill_between, args=[x[:,0], y16[:,0], y84[:,0]], kwargs={'alpha':0.2,'color':'k','label':'$\pm\sigma$'})
    ]

    default_graph(plots, "Measured Data")

x = np.concatenate((mock_x[:,np.newaxis], x), axis=0)
y50 = np.concatenate((mock_50[:,np.newaxis], y50), axis=0)
y16 = np.concatenate(((mock_50-mock_std)[:,np.newaxis], y16), axis=0)
std = np.concatenate((mock_std, std), axis=0)

# http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF
# http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.WhiteKernel.html#sklearn.gaussian_process.kernels.WhiteKernel
kernel = 1.0 * RBF(length_scale=1.0)
#kernel = 1.0 * Matern()
#kernel = 1.0 * RationalQuadratic()
#kernel = 1.0 * ExpSineSquared()
#kernel =  1.0 * Exponentiation(DotProduct(),2)


#https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/gaussian_process/gpr.py#L20
gp = GaussianProcessRegressor(kernel=kernel, alpha=std)

if GRAPH_PRIOR:
    x_ = np.linspace(0, x.max(), 100)[:, np.newaxis]
    y_, std_ = gp.predict(x_, return_std=True)

    plots = [
        Plot(func=plt.plot, args=[x,y50,'.'], kwargs={'color':'b', 'label':'Data Points'}),
        Plot(func=plt.plot, args=[x_,y_], kwargs={'color':'r','label':'Predicted Mean'}),
        Plot(func=plt.fill_between, args=[x_[:,0],y_-std_,y_+std_], kwargs={'alpha':.25, 'color':'r', 'label':'$\pm\sigma$'})
    ]

    default_graph(plots, 'Prior with params {}'.format(gp.kernel))

gp.fit(x,y50)
#print('Fitted Values: {}'.format(gp.kernel_))


if GRAPH_POSTERIOR:
    x_ = np.linspace(0, x.max(), 100)[:, np.newaxis]
    y_, std_ = gp.predict(x_, return_std=True)

    plots = [
        Plot(func=plt.plot, args=[x[x>=0],y50[x>=0],'.'], kwargs={'color':'b','label':'Data Points'}),
        Plot(func=plt.plot, args=[x_,y_], kwargs={'color':'r','label':'Predicted Mean'}),
        Plot(func=plt.fill_between, args=[x_[:,0],y_[:,0]-std_,y_[:,0]+std_], kwargs={'alpha':.25, 'color':'r', 'label':'$\pm\sigma$'})
    ]

    default_graph(plots, 'Fitted Model {}'.format(gp.kernel_))

if GRAPH_RECOVERED_MEAN:
    n_samples = 100

    samples = gp.sample_y(x[-orig_length:,:], n_samples) #returned shape (features, y, num_samples)
    mean = np.mean(samples, axis=2)

    plots = [
        Plot(func=plt.plot, args=[x[-orig_length:,:],y50[-orig_length:,:],'.'], kwargs={'color':'b','label':'Data Points'}),
        Plot(func=plt.plot, args=[x[-orig_length:,:],mean], kwargs={'color':'r','label':'Recovered Mean'})
    ]

    default_graph(plots, 'Mean From {} Samples'.format(n_samples))


if GENERATE_IMAGE:
    Re = 0.3
    OverRe = x.max()

    f = plt.figure()
    plt.gca().invert_yaxis()

    plt.title('Generated Disk Log-Scale')
    img, sample = generate_image(Re, OverRe, gp, return_sample=True)


    from PIL import Image
    r = Re/0.06
    #Q = 0.505218404406
    Q = 0.5
    b = np.sqrt(r**2 * Q)
    a = np.sqrt(r**2 / Q)

    print(a,b,r)

    to_origin = _translate(42, 42)
    scaled = _scale(a/r, b/r)
    recenter = _translate(-42, -42)
    trans = to_origin.dot(scaled).dot(recenter)
    print(trans)
    #http://stackoverflow.com/questions/17056209/python-pil-affine-transformation
    trans = tuple(np.linalg.inv(trans).flatten()[:6])
    #print(trans)


    tmp = Image.fromarray(img)
    tmp = tmp.transform(img.shape, Image.AFFINE, data=trans, resample=Image.NEAREST)
    tmp = np.asarray(tmp)
    plt.imshow(tmp, cmap='gray')

    with open('spheroid.json', 'w') as fs, open('disk.json', 'w') as fd:
        tmp_dict = {'img':dt._nmpy_encode(img)}
        json.dump(tmp_dict, fs)
        
        tmp_dict = {'img':dt._nmpy_encode(tmp)}
        json.dump(tmp_dict, fd)


    #tmp_ar = it.axis_ratio(tmp, np.ones_like(tmp, dtype=bool))
    #print(f'Given Axis Ratio:{Q} meaured:{tmp_ar}')



    plt.figure()

    plt.title('Generated Spheroid Log-Scale')
    plt.gca().invert_yaxis()
    plt.imshow(img, cmap='gray')
    #f.axes[0].get_xaxis().set_visible(False)
    #f.axes[0].get_yaxis().set_visible(False)

    x_ = np.linspace(0, x.max(), 100)[:, np.newaxis]
    y_, std_ = gp.predict(x_, return_std=True)

    plots = [
        Plot(func=plt.plot, args=[sample[0], sample[1]], kwargs={'label':'Sample'}),
        Plot(func=plt.plot, args=[x[-orig_length:,:], y50[-orig_length:,:], '.'], kwargs={'color':'b', 'label':'Data Points'}),
        Plot(func=plt.plot, args=[x_, y_], kwargs={'color':'r','label':'Predicted Mean'}),
        Plot(func=plt.fill_between, args=[x_[:,0], y_[:,0]-std_, y_[:,0]+std_], kwargs={'alpha':.25, 'color':'r', 'label':'$\pm\sigma$'})
    ]

    default_graph(plots, 'SBP Gaussian Process with params {}'.format(gp.kernel_))

if GRAPH_SAMPLES:
    samples = gp.sample_y(x[-orig_length:,:], GRAPH_SAMPLES)

    plots = [Plot(func=plt.plot, args=[x[-orig_length:,:], y50[-orig_length:,:],'.'], kwargs={'label':'Data Points'})]

    for i in range(GRAPH_SAMPLES):
        plots.append(Plot(func=plt.plot, args=[x[-orig_length:,:],samples[:,:,i]], kwargs={'alpha':0.2}))



    #plots.append(Plot(func=plt.fill_between, args=[x[-orig_length:,0], samples_mean[:,0]-samples_std[:,0], samples_mean[:,0]+samples_std[:,0]], kwargs={'alpha':0.5, 'color':'b'}))
    #plots.append(Plot(func=plt.errorbar, args=[x[-orig_length:,0], samples_std[:,0]], kwargs={}))

    default_graph(plots, '{} Generated Samples'.format(GRAPH_SAMPLES))

if GRAPH_SAMPLE_STD:
    num_samples = 100
    samples = gp.sample_y(x[-orig_length:,:], num_samples)
    samples_mean = np.mean(samples, axis=2)
    samples_std = np.std(samples, axis=2)

    samples_16 = []
    samples_84 = []

    for i in range(x[-orig_length:,:].shape[0]):
        samples_16.append(sorted(samples[i,0,:])[int(.15*num_samples)])
        samples_84.append(sorted(samples[i,0,:])[int(.84*num_samples)])

    plots = [
        Plot(func=plt.plot, args=[x[-orig_length:,0],y50[-orig_length:,0],'.'], kwargs={'color':'b','label':'Median'}),
        Plot(func=plt.fill_between, args=[x[-orig_length:,0], y16[-orig_length:,0], y84[-orig_length:,0]], kwargs={'alpha':0.2,'color':'b','label':'$\pm\sigma$'}),
        Plot(func=plt.fill_between, args=[x[-orig_length:,0], y50[-orig_length:,0]-std[-orig_length:], y50[-orig_length:,0]+std[-orig_length:]], kwargs={'alpha':0.2,'color':'g','label':'$\pm\sigma_n$ altered'}),
        Plot(func=plt.errorbar, args=[x[-orig_length:,:], samples_mean], kwargs={'fmt':'.','yerr':samples_std,'color':'r','label':'Sample Mean'}),

        #Plot(func=plt.plot, args=[x[-orig_length:,0], samples_16, '--'], kwargs={'label':'16','color':'r'}),
        #Plot(func=plt.plot, args=[x[-orig_length:,0], samples_84, '--'], kwargs={'label':'84','color':'r'})
    ]

    default_graph(plots, "Dispersion of Samples")


plt.show()






