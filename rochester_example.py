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
Creation Date:  Sat Apr 14 17:00 2018
Filename:       rochester_example.py

"""
from __future__ import print_function
import numpy as np

from HMM import HMM

"""
Model described in http://www.cs.rochester.edu/u/james/CSC248/Lec11.pdf
"""

A = np.array([[0.60, 0.40],
              [0.30, 0.70]])

B = np.array([[0.3, 0.4, 0.3],
              [0.4, 0.3, 0.3]])


seq = ['R', 'W', 'B', 'B']

model = HMM(A, B, pi0=np.array([0.8, 0.2]),
            states=['S1', 'S2'],
            emissions=['R', 'W', 'B'])

res = model.forward_backward(seq)

print(model)
model.print_matrix(res)
