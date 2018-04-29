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
Filename:       HMM.py

"""
from __future__ import print_function

from tabulate import tabulate
import numpy as np
import pdb


class HMM(object):

    def __init__(self, A, B, pi0=None, states=None, emissions=None):
        """
        :param A: Transition matrix of shape (n, n) (n = number of states)
        :param B: Emission matrix of shape (n, b) (b = number of outputs)
        :param pi0: Initial State Probability vector of size n, leave blank for uniform probabilities
        :param states: State names/labels as list
        :param emissions: Emission names/labels as list
        :raises: Assertion error on bad input parameters
        """
        self.A = A
        self.B = B
        self.n_states = A.shape[0]
        self.n_emissions = B.shape[1]
        self.states = states
        self.emissions = emissions
        self.pi0 = pi0

        n = self.n_states

        if pi0 is None:
            self.pi0 = np.full(n, 1.0 / n)

        if states is None:
            self.states = [chr(ord('A') + i) for i in range(n)]

        if emissions is None:
            self.emissions = [str(i) for i in range(self.n_emissions)]

        assert A.shape[0] == A.shape[1]
        assert B.shape[0] == self.n_states
        assert self.pi0.shape[0] == self.n_states

        for row in A:
            assert sum(row) == 1.0
        for row in B:
            assert sum(row) == 1.0

    def __repr__(self):
        """
        :return: String representation of model
        """
        s = "Transition Matrix (A):\n{}\n\n".format(self._matrix_repr(self.A))
        s += "Emission Matrix (B):\n{}\n\n".format(self._matrix_repr(self.B, headers=self.emissions))
        s += "Initial State Vector (pi_0):\n{}\n\n".format(self._matrix_repr(self.pi0))
        return s

    def forward_algorithm(self, seq):
        """
        Apply forward algorithm to calculate probabilities of seq

        :param seq: Observed sequence to calculate probabilities upon
        :return: Alpha matrix with 1 row per time step
        """

        A = self.A
        B = self.B
        pi0 = self.pi0
        n = self.n_states

        # infer time steps
        T = len(seq)

        # create a T by n empty matrix
        # Row 't' will contain forward probabilities
        # for each state
        M = np.zeros((T, n))

        # calculate forward probabilities
        # for t = 1, based on initial state vector
        for i in range(n):
            M[0, i] = B[i, seq[0]] * pi0[i]

        # calculate probabilities for t > 1
        for t in range(1, T):
            # loop over each state k:0->n
            for k in range(n):
                tmp = 0
                # loop over each term in summation
                # i:0->n
                for i in range(n):
                    tmp += A[i, k] * M[t - 1, i]
                # finally, multiply term with emission probability of
                # observed output at time t
                M[t, k] = tmp * B[k, seq[t]]

        return M

    def backward_algorithm(self, seq):
        """
        Apply backward algorithm to calculate probabilities of seq

        :param seq: Observed sequence to calculate probabilities upon
        :return: Beta matrix with 1 row per timestep
        """

        A = self.A
        B = self.B
        pi0 = self.pi0
        n = self.n_states

        # infer time steps
        T = len(seq)

        # create a T by n empty matrix
        # Row 't' will contain forward probabilities
        # for each state
        M = np.zeros((T, self.n_states))

        # calculate probabilities for all time steps
        # starting from T-1, down to 0
        for t in range(T - 1, -1, -1):

            # loop over each state k to calculate probability
            # for time t
            for k in range(self.n_states):
                tmp = 0
                # loop over each summation term
                for j in range(n):
                    emission_prob = 1.0 if t + 1 == T else B[j, seq[t + 1]]
                    beta_t = 1.0 if t + 1 == T else M[t + 1, j]
                    tmp += A[k, j] * emission_prob * beta_t
                M[t, k] = tmp

        # calculate beta_0 vector separately
        # we don't want it interfering with our
        # calculations/indexing later
        b0_vector = np.zeros((1, self.n_states))
        for k in range(n):
            b0_vector[0, k] = pi0[k] * B[k, seq[0]] * M[0, k]

        return M, b0_vector

    def alphaTsum(self, M):
        """
        Returns alpha_T sum

        :param M: Input matrix
        :return: sum of probabilities at time = T (final row)
        """
        return sum(M[-1])

    def beta0sum(self, b0):
        """
        Returns beta_0 sum

        :param M: beta_0 vector
        :return: sum of probabilities at time = 0 (first row)
        """
        return sum(b0[0])

    def forward_backward(self, seq):
        """
        Applies forward-backward algorithm to seq

        :param seq: Observed sequence to calculate probabilities upon
        :return: Matrix M containing state probabilities for each timestamp
        :raises: ValueError on bad sequence
        """

        # convert sequence to integers
        if all(isinstance(i, str) for i in seq):
            seq = [self.emissions.index(i) for i in seq]

        n = self.n_states
        T = len(seq)

        alpha_matrix = self.forward_algorithm(seq)
        beta_matrix, b0_vector = self.backward_algorithm(seq)

        M = np.zeros((T, self.n_states))

        # we could also use beta_0 = self.beta0sum(b0_vector)
        alpha_T = self.alphaTsum(alpha_matrix)
        for t in range(T):
            for i in range(n):
                M[t, i] = (alpha_matrix[t, i] * beta_matrix[t, i]) / alpha_T

        return M

    def viterbi(self, seq):
        """
        :param seq: observed sequence
        :return: Viterbi path as python list
        """
        # convert sequence to integers
        if all(isinstance(i, str) for i in seq):
            seq = [self.emissions.index(i) for i in seq]
        
        A = self.A
        B = self.B
        pi0 = self.pi0
        n = self.n_states
        state_names = self.states
        emission_names = self.emissions

        pass
        
        

    def _matrix_repr(self, M, headers=None):
        """
        Tabulated form of matrix representation

        :param M: Matrix to print
        :param headers: Optional headers for columns, default is state names
        :return: tabulated encoding of input matrix
        """
        headers = headers or self.states

        if M.ndim > 1:
            headers = [' '] + headers
            data = [['t={}'.format(i + 1)] + [j for j in row] for i, row in enumerate(M)]
        else:
            data = [[j for j in M]]

        return tabulate(data, headers, tablefmt="grid", numalign="right")

    def print_matrix(self, M):
        """
        Print matrix in tabular form

        :param M: Matrix to print
        :return: None
        """
        print(self._matrix_repr(M))
