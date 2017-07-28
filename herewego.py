import DataTools as dt
import ImageTools as it
import numpy as np
import matplotlib.pyplot as plt

def to_bin_collection(s_coll):
    s_rad_bin = []

    for s in s_coll:
        src_map = s[2]==s[0]
        # returns a dictionary of R/Re->I/Ie
        srad = it.radial_bin_image(s[1], src_map, re_limit=4.0, re_normed=True, ie_normed=True)
        s_rad_bin.append(srad)

    return s_rad_bin

v_sph = dt.spheroids(bands='v', log_scale=False, orig=True)
z_sph = dt.spheroids(bands='z', log_scale=False, orig=True)
h_sph = dt.spheroids(bands='h', log_scale=False, orig=True)
j_sph = dt.spheroids(bands='j', log_scale=False, orig=True)

v_sph_rad = to_bin_collection(v_sph)
z_sph_rad = to_bin_collection(z_sph)
h_sph_rad = to_bin_collection(h_sph)
j_sph_rad = to_bin_collection(j_sph)


if True:
    for b, d in zip(['v', 'z', 'h', 'j'], [v_sph, z_sph, h_sph, j_sph]):
        plt.figure()
        plt.title(f'{b} Central Value')

        vals = []
        for coll in d:
            src_map = coll[2]==coll[0]
            vals.append(np.max(coll[1][src_map]))

        plt.hist(vals)

if False:
    for b, d in zip(['V', 'Z', 'H', 'J'], [v_sph_rad, z_sph_rad, h_sph_rad, j_sph_rad]):
        plt.figure()
        plt.xlabel('$R/R_e$')
        plt.ylabel('$I/I_e$')
        plt.title(f'{b} SBP')
        for coll in d:
            xs, ys = [], []
            for r in sorted(coll.keys()):
                xs.append(r)
                ys.append(coll[r])

            plt.plot(xs, ys)

if False:
    for b, d in zip(['v', 'z', 'h', 'j'], [v_sph_rad, z_sph_rad, h_sph_rad, j_sph_rad]):
        r_vals = []
        i_vals = []
        for val in d:
            for r in val.keys():
                r_vals.append(r)
                i_vals.append(val[r])

        plt.figure()
        plt.hist(r_vals, bins=5000, range=(0, 5))
        plt.title(f'{b} band $I/I_e$ Values Histogram')
        plt.xlabel('$I/I_e$')
        plt.ylim((0, 250))
        plt.xlim((-0.5, 5.5))


        i_tmp = np.array(i_vals)

        plt.figure()
        plt.hist(i_tmp[np.isnan(i_tmp)==False], bins=5000, range=(0, 10))
        plt.title(f'{b} band $R/R_e$ Values Histogram')
        plt.xlabel('$R/R_e$')
        plt.ylim((0, 600))
        plt.xlim((-0.5, 10))


plt.show()
