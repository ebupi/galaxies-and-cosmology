#!/usr/bin/env python
#
# sfd.py
#
# Read the SFD (1998) dust reddening map.
#
# Expects the files SFD_dust_4096_ngp.fits and SFD_dust_4096_sgp.fits to be
# in a directory, provided to SFDQuery as the map_dir argument. Googling these
# file names should be sufficient to find them.
#

import os
import h5py
import numpy as np

import astropy.wcs as pywcs
import astropy.io.fits as fits
from scipy.ndimage import map_coordinates

class SFDQuery():
    def __init__(self, map_dir):
        self.data = {}

        base_fname = os.path.join(map_dir, 'SFD_dust_4096')

        for pole in ['ngp', 'sgp']:
            fname = '{}_{}.fits'.format(base_fname, pole)
            with fits.open(fname) as hdulist:
                self.data[pole] = hdulist[0].header, hdulist[0].data

    def query(self, l, b, order=1):
        l = np.asarray(l)
        b = np.asarray(b)

        if l.shape != b.shape:
            raise ValueError('l.shape must equal b.shape')

        out = np.zeros_like(l, dtype='f4')

        for pole in ['ngp', 'sgp']:
            m = (b >= 0) if pole == 'ngp' else b < 0

            if np.any(m):
                header, data = self.data[pole]
                wcs = pywcs.WCS(header)

                if not m.shape: # Support for 0-dimensional arrays (scalars). Otherwise it barfs on l[m], b[m]
                    x, y = wcs.wcs_world2pix(l, b, 0)
                    out = map_coordinates(data, [[y], [x]], order=order, mode='nearest')[0]
                    continue

                x, y = wcs.wcs_world2pix(l[m], b[m], 0)
                out[m] = map_coordinates(data, [y, x], order=order, mode='nearest')

        return out

    def __call__(self, *args, **kwargs):
        return self.query(*args, **kwargs)
