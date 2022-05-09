# Refactoring the QAOA implementation
# Author: Edmund Dable-Heath
"""
    Refactoring the QAOA implementation of the basis reduction written by Rocky.
"""

import numpy as np
import scipy as sp
import qutip as qp
import itertools as it
import sys
np.set_printoptions(threshold=sys.maxsize)


class QAOA:

    """
        Main class for QAOA model
    """

    def __init__(self, **pars):
        """
        parameter list:
            - basis: lattice basis, int-(m, m)-ndarray
            - integer_range: integer range to search over, int
            - iters: max iteration, int
            - fix_Hp: choose whether to fix Hp or not, bool
            - fix_beta: choose whether to fix beta or not, bool
        """
        # Problem parameters
        self.basis = qp.Qobj(pars['basis'])
        self.dimension = pars['basis'].shape[0]
        self.int_range = pars['integer_range']
        self.num_qubits = 6 # todo: compute this from integer range
        if 'fix_Hp' not in pars:
            self.fix_Hp = False
        else:
            self.fix_Hp = pars['fix_Hp']
        if 'fix_beta' not in pars:
            self.fix_beta = False
        else:
            self.fix_beta = pars['fix_beta']

        # Simulation parameters
        self.iters = pars['iters']
        self.current_iter = 0
        self.step_track = []
        self.basis_track = []
        self.basis_track.append(self.basis)
        self.magnitude_track = []
        self.avg_norm_track = np.zeros(self.iters)
        self.norm_track = np.zeros(self.iters, self.dimension)

        self.step_track.append(0)
        self.norm_track[0] = self.norms()
        self.avg_norm_track[0] = self.avg_norm()

    def to_vec(self):
        """
            Return array form of basis (in column form) from qutip object.
        :return: basis, int-(m, m)-np.array
        """
        return np.array(self.basis.data.todense().real.T, dtype=int)

    def avg_norm(self):
        """
            Return avg norm of basis vectors for current basis.
        :return: mean of norm of basis vectors, float
        """
        return np.round(np.mean(np.linalg.norm(self.to_vec(), axis=1)), 2)

    def norms(self):
        """
            Return norms of current basis.
        :return: (m, )-ndarray of norms, float
        """
        return np.round(np.linalg.norm(self.to_vec(), axis=1), 2)

    def gramm(self):
        """
            Compute the Gramm matrix for the current basis
        :return: gramm matrix, int-(m, m)-ndarray
        """
        if self.fix_Hp:
            return self.basis_track[0].dag() * self.basis_track[0]
        else:
            return self.basis.dag() * self.basis

    def pauli_z(self, i):
        """
            pauli_z operator acting on ith position.
        :param i: position parameter, int
        :return: qutip tensor for pauli_z operator acting on ith position
        """
        string = [qp.qeye(2)] * self.num_qubits
        string[i] = qp.sigmaz()
        return qp.tensor(string)

    def pauli_x(self, i):
        """
            pauli_x operator acting on the ith position.
        :param i: position parameter, int
        :return: qutip tensor for pauli_x operator acting on ith position
        """
        string = [qp.qeye(2)] * self.num_qubits * self.dimension
        string[i] = qp.sigmax()
        return qp.tensor(string)

    def bin_encoding(self):
        """
            Binary encoding of problem in qubits
        :return: state represented as qutip tensor
        """
        x = [2**(k-1)*self.pauli_z(k) for k in range(self.num_qubits)]
        return sum(x) + qp.tensor([qp.qeye(2)]*self.num_qubits)/2

    def problem_ham(self):
        """
            Computing the problem Hamiltonian for the VQE
        :return: qutip tensor representation of Hamiltonian
        """
        x = []
        gramm = self.gramm()
        for i in range(self.dimension):
            for j in range(self.dimension):
                string_1 = [qp.tensor([qp.qeye(2)]*self.num_qubits)] * self.dimension
                string_2 = [qp.tensor([qp.qeye(2)]*self.num_qubits)] * self.dimension
                string_1[i] = self.bin_encoding()
                string_2[j] = self.bin_encoding()
                QiQj = qp.tensor(string_1) * qp.tensor(string_2)
                xx = np.real(gramm[i][0][j]) * QiQj
                x.append(xx)
        return sum(x)

    def problem_ham_i(self):
        """
            Set of problem Hamiltonians for given set.
        :return: list of qutip tensors
        """
        estates = self.problem_ham().eigenstates()[1]
        return [self.problem_ham()*i*i.dag() for i in estates]

    def driver_ham(self):
        """
            Computing the driver (ansatze) hamiltonian for the VQE
        :return: qutip tensor representation of Hamiltonian
        """
        return sum([self.pauli_x(i) for i in range(self.num_qubits*self.dimension)])

    def psi_0(self):
        """
            Compute basis state.
        :return: qutip eigenstates
        """
        string = [qp.qeye(2)] * self.num_qubits * self.dimension
        return sum(qp.tensor(string).eigenstates()[1]).unit()




if __name__ == "__main__":
    basis = np.array([[1, 2, -4, 0],
                      [-1, -4, -3, 5],
                      [-1, -8, 6, 9],
                      [9, -2, 2, 9]])
    int_range = 6
    parameters = {
        'basis': basis,
        'integer_range': int_range,
        'fix_Hp': False,
        'fix_beta': False
    }
