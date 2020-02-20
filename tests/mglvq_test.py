#!/usr/bin/python3
"""
Tests the median generalized learning vector quantization implementation
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
import proto_dist_ml.mglvq as mglvq

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestMGLVQ(unittest.TestCase):

    def test_mgvlq1(self):
        # create the simplest possible test case of two Gaussian clusters
        # with slight overlap and binary classification
        K = 2
        m = 100
        X = np.zeros((K*m, 2))
        y = np.zeros(K*m)
        W_expected = []
        y_W = []
        for k in range(K):
            # get the location of the mean
            theta = k * 2 * np.pi / K
            mu = 1. * np.array([np.cos(theta), np.sin(theta)])
            W_expected.append(mu)
            y_W.append(k)
            # generate data
            X[k*m:(k+1)*m, :] = np.random.randn(m, 2) + mu
            y[k*m:(k+1)*m] = k
        W_expected = np.stack(W_expected, axis=0)
        y_W = np.stack(y_W, axis=0)
        # compute all pairwise Euclidean distances
        D = cdist(X, X)
        # set up a mglvq model
        model = mglvq.MGLVQ(1)
        # train it
        model.fit(D, y)
        # check the result
        W_actual = X[model._w, :]
        np.testing.assert_allclose(W_actual, W_expected, atol=1.)
        # ensure high classification accuracy
        self.assertTrue(model.score(D, y) > 0.8)

    def test_mgvlq2(self):
        # create a rather simple test data set of K Gaussian clusters in a circle,
        # with a new label for each of them
        K = 6
        m = 100
        X = np.zeros((K*m, 2))
        y = np.zeros(K*m)
        W_expected = []
        y_W = []
        for k in range(K):
            # get the location of the mean
            theta = k * 2 * np.pi / K
            mu = 3. * np.array([np.cos(theta), np.sin(theta)])
            W_expected.append(mu)
            y_W.append(k)
            # generate data
            X[k*m:(k+1)*m, :] = np.random.randn(m, 2) + mu
            y[k*m:(k+1)*m] = k
        W_expected = np.stack(W_expected, axis=0)
        y_W = np.stack(y_W, axis=0)
        # compute all pairwise Euclidean distances
        D = cdist(X, X)
        # set up a mglvq model
        model = mglvq.MGLVQ(1)
        # train it
        model.fit(D, y)
        # check the result
        W_actual = X[model._w, :]
        np.testing.assert_allclose(W_actual, W_expected, atol=1.)
        # ensure high classification accuracy
        self.assertTrue(model.score(D, y) > 0.8)

    def test_mgvlq3(self):
        # create a rather simple test data set of K Gaussian clusters in a circle,
        # with interleaved labels
        K = 6
        m = 100
        X = np.zeros((K*m, 2))
        y = np.zeros(K*m)
        W_expected = []
        y_W = []
        for k in range(K):
            # get the location of the mean
            theta = k * 2 * np.pi / K
            mu = 3. * np.array([np.cos(theta), np.sin(theta)])
            W_expected.append(mu)
            y_W.append(k % 2)
            # generate data
            X[k*m:(k+1)*m, :] = np.random.randn(m, 2) + mu
            y[k*m:(k+1)*m] = k % 2
        W_expected = np.stack(W_expected, axis=0)
        y_W = np.stack(y_W, axis=0)
        # compute all pairwise Euclidean distances
        D = cdist(X, X)
        # set up a mglvq model
        model = mglvq.MGLVQ(int(K / 2))
        # train it
        model.fit(D, y)
        # check the result
        W_actual = X[model._w, :]
        for l in range(2):
            # consider the prototypes for each label and ensure
            # that for each actual prototype there is at least one
            # very close expected prototype
            D_l = cdist(W_actual[model._y == l, :], W_expected[y_W == l, :])
            np.testing.assert_allclose(np.min(D_l, axis=1), np.zeros(len(D_l)), atol=1.)
        # ensure high classification accuracy
        self.assertTrue(model.score(D, y) > 0.8)

    def test_mgvlq4(self):
        # create a data set with four clusters placed in a square,
        # three of which belong to one class, and one of which belongs
        # to the other class
        m = 100
        X = np.zeros((4*m, 2))
        y = np.zeros(4*m)
        for k in range(4):
            # get the location of the mean
            theta = (k + 0.5) * 2 * np.pi / 4
            mu = 3. * np.array([np.cos(theta), np.sin(theta)])
            # generate data
            X[k*m:(k+1)*m, :] = np.random.randn(m, 2) + mu
            y[k*m:(k+1)*m] = 0 if k == 0 else 1
        # compute all pairwise Euclidean distances
        D = cdist(X, X)
        # set up a mglvq model
        model = mglvq.MGLVQ(2)
        # train it
        model.fit(D, y)
        # check the result
        self.assertTrue(len(model._loss) > 1)
        W_actual = X[model._w, :]
        # ensure high classification accuracy
        self.assertTrue(model.score(D, y) > 0.8)

if __name__ == '__main__':
    unittest.main()
