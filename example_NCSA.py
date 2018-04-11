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
Filename:       example_NCSA.py

"""
from __future__ import print_function
import numpy as np

from HMM import HMM

"""
NCSA Example from Homework 5
"""

A = np.array([[0.25, 0.75, 0.00],
              [0.00, 0.25, 0.75],
              [0.00, 0.00, 1.00]])

B = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 0.0]])

seq = ['PS', 'SI', 'PS']

model = HMM(A, B,
            states=['NA', 'AP', 'AC'],
            emissions=['PS', 'SI'])

print(model)

res = model.forward_backward(seq)
model.print_matrix(res)
