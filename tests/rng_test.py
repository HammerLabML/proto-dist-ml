#!/usr/bin/python3
"""
Tests the relational neural gas implementation
"""

# Copyright (C) 2019
# Benjamin Paaßen
# AG Machine Learning
# Bielefeld University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy as np
from scipy.spatial.distance import cdist
import proto_dist_ml.rng as rng

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestRNG(unittest.TestCase):

    def test_rng(self):
        # create a rather simple test data set of K Gaussian clusters in a circle
        K = 6
        m = 100
        X = np.zeros((K*m, 2))
        W_expected = []
        for k in range(K):
            # get the location of the mean
            theta = k * 2 * np.pi / K
            mu = 3. * np.array([np.cos(theta), np.sin(theta)])
            W_expected.append(mu)
            # generate data
            X[k*m:(k+1)*m, :] = np.random.randn(m, 2) + mu
        W_expected = np.stack(W_expected, axis=0)
        # compute all pairwise Euclidean distances
        D = cdist(X, X)
        # set up a relational neural gas model
        model = rng.RNG(K)
        # train it
        model.fit(D)
        # check the result
        W_actual = model._Alpha.dot(X)
        idxs = model.predict(cdist(W_expected, X))
        np.testing.assert_allclose(W_actual[idxs, :], W_expected, atol=0.5)

if __name__ == '__main__':
    unittest.main()
