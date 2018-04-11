"""
Permission to use, copy, modify, and distribute this software and its
documentation for any purpose, without fee, and without written agreement is
hereby granted, provided that the above copyright notice and the following
two paragraphs appear in all copies of this software.

IN NO EVENT SHALL THE AUTHOR OR THE UNIVERSITY OF ILLINOIS BE LIABLE TO
ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES ARISING OUT  OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION,
EVEN IF THE AUTHOR AND/OR THE UNIVERSITY OF ILLINOIS HAS BEEN ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

THE AUTHOR AND THE UNIVERSITY OF ILLINOIS SPECIFICALLY DISCLAIM ANY
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE
PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND NEITHER THE AUTHOR NOR
THE UNIVERSITY OF ILLINOIS HAS ANY OBLIGATION TO PROVIDE MAINTENANCE,
SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS."

Author:         Lavin Devnani
Version:        1
Creation Date:  Wed Apr 11 18:00 2018
Filename:       test_HMM.py

"""

from __future__ import print_function

import numpy as np
import pytest

from HMM import HMM


@pytest.fixture(scope="module")
def ncsa_model():
    """
    Model described in hw5

    :return: Transition, Emission and Expected Matrices
    """
    A = np.array([[0.25, 0.75, 0.00],
                  [0.00, 0.25, 0.75],
                  [0.00, 0.00, 1.00]])

    B = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0]])

    expected = np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])

    seq = ['PS', 'SI', 'PS']

    model = HMM(A, B,
                states=['NA', 'AP', 'AC'],
                emissions=['PS', 'SI'])

    return model, seq, expected


@pytest.fixture(scope="module")
def gene_model():
    """
    Model described in lecture 17

    :return: Transition, Emission and Expected Matrices
    """
    A = np.array([[0.70, 0.30],
                  [0.30, 0.70]])

    B = np.array([[0.1, 0.1, 0.4, 0.4],
                  [0.3, 0.3, 0.2, 0.2]])

    expected = np.array([[0.20246232, 0.79753768],
                         [0.24787943, 0.75212057],
                         [0.63229294, 0.36770706],
                         [0.74173451, 0.25826549],
                         [0.74173451, 0.25826549],
                         [0.63229294, 0.36770706],
                         [0.24787943, 0.75212057],
                         [0.20246232, 0.79753768]])

    seq = ['A', 'C', 'G', 'T', 'G', 'G', 'C', 'A']

    model = HMM(A, B,
                states=['G', 'NG'],
                emissions=['A', 'C', 'G', 'T'])

    return model, seq, expected


@pytest.fixture(scope="module")
def example_model():
    """
    Model described in http://www.cs.rochester.edu/u/james/CSC248/Lec11.pdf

    :return: Transition, Emission and Expected Matrices
    """
    A = np.array([[0.60, 0.40],
                  [0.30, 0.70]])

    B = np.array([[0.3, 0.4, 0.3],
                  [0.4, 0.3, 0.3]])

    expected = np.array([[0.0324, 0.0297],
                         [0.09, 0.09],
                         [0.3, 0.3],
                         [1.0, 1.0]])

    seq = ['R', 'W', 'B', 'B']

    model = HMM(A, B,
                states=['S1', 'S2'],
                emissions=['R', 'W', 'B'])

    return model, seq, expected


def check_float_array(a1, a2, tolerance=1e-6):
    assert a1.shape == a2.shape

    f1 = np.ndarray.flatten(a1)
    f2 = np.ndarray.flatten(a1)

    for i in range(len(f1)):
        assert abs(f1[i] - f2[i]) <= tolerance


def test_forward_backward():
    tests = [ncsa_model, gene_model, example_model]
    for model in tests:
        m, s, e = model()
        res = m.forward_backward(s)
        check_float_array(res, e)
