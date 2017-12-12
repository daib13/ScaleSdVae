import pickle
import matplotlib.pyplot as plt
import numpy as np


def load_data(filename):
    fid = open(filename, 'rb')
    x = pickle.load(fid)
    fid.close()
    return x


dh_d = load_data('dh_d.bin')
dh_d = np.asarray(dh_d)
dw_o = load_data('dw_o.bin')
dz = load_data('dz.bin')

w_o = load_data('w_o_norm.bin')
sd_z = load_data('sd_z_mean.bin')

red = [200.0/255.0, 37.0/255.0, 4.0/255.0]
blue = [38.0/255.0, 121.0/255.0, 199.0/255.0]
purple = [127.0/255.0, 75.0/255.0, 161.0/255.0]

for i in range(30):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.plot(dh_d[10:] * 1000, color=purple)
#    ax1.axis([0, 10000, 0, 0.2])
#    plt.figlegend((line_dh_d), (r'$||\frac{d\mathcal{L}}{dh_d}||_2$'), 'upper right')

    ax2.plot(w_o[10:, i], color=blue)
    ax2.plot(sd_z[10:, i], color=red)
    ax2.axis([0, 10000, 0, 10])
#    plt.figlegend((line_w_o, line_sd_z), (r'$||w_{o,\cdot i}||_2^2$', r'$\bar{\sigma_{z,i}}$'), 'upper right')

    ax3.plot(dw_o[10:, i], color=blue)
    ax3.axis([0, 10000, 0, 5])

    ax4 = ax3.twinx()
    ax4.plot(dz[10:, i] * 1000, color=red)
    ax4.axis([0, 10000, 0, 0.05])

    plt.show()
