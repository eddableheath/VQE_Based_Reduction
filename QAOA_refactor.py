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
import construct_basis as cb
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
            - gamma: gamma parameter
            - beta: beta parameter
            - tlist: tlist...?
        """
        # Problem parameters
        self.basis = qp.Qobj(pars['basis'])
        self.dimension = pars['basis'].shape[0]
        self.int_range = pars['integer_range']
        self.num_qubits = 2  # todo: compute this from integer range
        if 'fix_Hp' not in pars:
            self.fix_Hp = False
        else:
            self.fix_Hp = pars['fix_Hp']
        if 'fix_beta' not in pars:
            self.fix_beta = False
        else:
            self.fix_beta = pars['fix_beta']

        # Optimisation parameters
        self.gamma = pars['gamma']
        self.beta = pars['beta']
        self.tlist = pars['tlist']

        # Simulation parameters
        self.iters = pars['iters']
        self.current_iter = 0
        self.step_track = []
        self.basis_track = []
        self.basis_track.append(self.basis)
        self.magnitude_track = []
        self.avg_norm_track = np.zeros(self.iters)
        self.norm_track = np.zeros((self.iters, self.dimension))

        self.step_track.append(0)
        self.norm_track[0] = self.norms()
        self.avg_norm_track[0] = self.avg_norm()

        # initial state
        self.psi_0 = self.set_psi_0()
        self.current_psi = self.psi_0

        # problem hamiltonians and propagators
        self.initial_problem_ham = self.update_problem_ham()
        self.problem_ham = self.update_problem_ham()
        self.problem_prop = self.update_problem_prop()
        self.problem_prop_min = self.update_problem_prop(minimum=True)
        self.driver_ham = self.compute_driver_ham()
        self.driver_prop = self.set_driver_prop()
        self.driver_prop_min = self.set_driver_prop(minimum=True)

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

    def update_problem_ham(self):
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
        estates = self.problem_ham.eigenstates()[1]
        return [self.problem_ham*i*i.dag() for i in estates]

    def update_problem_prop(self, minimum=False):
        """
            Update the problem propagator
        :param minimum: return the minimal version
        :return: qutip propagator
        """
        if not minimum:
            return qp.propagator(self.gamma*self.problem_ham, self.tlist)
        else:
            return qp.propagator(self.gamma*self.problem_ham, self.tlist, unitary_mode='single')

    def compute_driver_ham(self):
        """
            Computing the driver (ansatze) hamiltonian for the VQE
        :return: qutip tensor representation of Hamiltonian
        """
        return sum([self.pauli_x(i) for i in range(self.num_qubits*self.dimension)])

    def set_psi_0(self):
        """
            Compute initial state.
        :return: qutip eigenstates
        """
        string = [qp.qeye(2)] * self.num_qubits * self.dimension
        return sum(qp.tensor(string).eigenstates()[1]).unit()

    def set_driver_prop(self, minimum=False):
        """
            Compute driver propagator
        :param minimum: return the minimal version
        :return: qutip propagator
        """
        if minimum:
            return qp.propagator(self.beta*self.driver_ham, self.tlist, unitary_mode='single')
        else:
            return qp.propagator(self.beta*self.driver_ham, self.tlist)

    def psi(self, minimum=None):
        """
            Compute current state.
        :param minimum:
        :return: propagated stated, qutip eigenstate
        """
        if self.fix_beta:
            arg = np.round((.25*np.pi)/self.tlist[1])
            if minimum is not None:
                return self.problem_prop_min[arg]*self.driver_prop_min[minimum]*self.psi_0
            else:
                return self.driver_prop[arg]*self.problem_prop*self.psi_0
        else:
            if minimum is not None:
                return self.driver_prop_min[minimum]*self.problem_prop_min[minimum]*self.psi_0
            else:
                return self.driver_prop*self.problem_prop*self.psi_0

    def expectation(self, minimum=None):
        """
            Compute expectation value
        :param minimum: given minimum to run on
        :return: expectation value
        """
        if minimum is not None:
            return qp.expect(self.problem_ham_i(), self.psi(minimum=minimum))
        else:
            return qp.expect(self.problem_ham, self.psi())

    def find_states(self):
        """
            Find states....?
        :return: states...?
        """
        expec = self.expectation()
        minimum = np.argmin(expec)
        exp_min = self.expectation(minimum=minimum)

        prob_ham_estates = self.problem_ham.eigenstates()
        prob_min = np.zeros(len(exp_min))
        for i in range(prob_min.shape[0]):
            if prob_ham_estates[0][i] == 0:
                prob_min[i] = 0
            else:
                prob_min[i] = exp_min[i] / prob_ham_estates[0][i]

        E = prob_ham_estates.tolist()

        for x in range(0, 3):
            for i in range(prob_min.shape[0]):
                if i < prob_min.shape[0] - 1:
                    for j in range(i+1, i+2):
                        if E[i] == E[j]:
                            prob_min[i] += prob_min[j]
                            np.delete(prob_min, j)
                            E.pop(j)
                        else:
                            pass
        prob_min[0] = 1 - np.sum(prob_min)
        return prob_min, E

    def solve(self, evalue):
        """
            Solve for integer values for given eigenvalue, agnostic to dimension just brute force searches over integers
            for element with corresponding norm?

            Todo: there must be a smarter way to solve for this?
        :param evalue: given eigenvalue
        :return: integer vector
        """
        b = self.to_vec()
        max_range = 2**self.num_qubits

        for int_guess in it.combinations_with_replacement(np.arange(-max_range, max_range-1), self.dimension):
            if np.round(np.linalg.norm(np.dot(b.T, int_guess))**2) == evalue:
                return np.dot(b.T, int_guess)
            else:
                pass
        return None

    def sim_step(self):
        """
            Step of the montecarlo simulation
        :return: updates simulation parameters
        """
        b = self.to_vec()
        basis_E = np.linalg.norm(basis, axis=1)
        prob, evalues = self.find_states()
        det = np.linalg.det(basis)
        chosen_E = 9e10
        det_new = 0
        basis_new = np.zeros((self.dimension, self.dimension))
        rand_vec = None
        while max(basis_E) <= chosen_E or \
                np.abs(np.round(det)) != np.abs(np.round(det_new)) or \
                (basis_new == b).all() or \
            rand_vec is None:
            self.current_iter += 1
            if self.current_iter >= self.iters:
                print(f'max step achieved: {self.iters}')
                exit()
            else:
                rand_ind = np.random.choice(np.arange(prob.shape[0]), p=prob)
                chosen_E = evalues[rand_ind]
                chosen_P = prob[rand_ind]
                rand_vec = self.solve(chosen_E)

                if rand_vec is None:
                    rand_vec = [0]*self.basis.dims[0][0]
                    chosen_E = 9e10
                    print('solution is None')

                prim_vec = cb.check_for_primitivity(rand_vec)
                new_basis = cb.construct_basis(prim_vec, b)
                if np.round(np.linalg.det(new_basis)) == np.round(np.linalg.det(b)):
                    b = new_basis
                else:
                    print('basis construnction failed')

        self.basis = qp.Qobj(np.transpose(b).tolist())
        total_E = np.linalg.norm(b)
        avg_norm = np.round(self.avg_norm(), 2)

        self.norm_track[self.current_iter] = self.norms()
        self.basis_track[self.current_iter] = self.basis
        self.magnitude_track.append(total_E)

        print(f'iteration {self.current_iter} ----------------------------------------------------')
        print(f'selected state: {chosen_E}')
        print(f'with probability: {chosen_P}')
        print(f'norms: {self.norm_track[self.current_iter]}')
        print(f'average norms: {self.avg_norm_track[self.current_iter]}')
        print(f'new basis: {self.basis}')
        self.problem_ham = self.update_problem_ham()
        self.problem_prop = self.update_problem_prop()
        self.problem_prop_min = self.update_problem_prop(minimum=True)

    def iterate(self):
        while self.current_iter <= self.iters:
            self.sim_step()


if __name__ == "__main__":
    basis = np.array([[3, 0, 15, -12],
                      [0, 4, 3, 8],
                      [28, -18, 9, 8],
                      [0, 0, 3, -4]])
    int_range = 2
    parameters = {
        'basis': basis,
        'integer_range': int_range,
        'fix_Hp': False,
        'fix_beta': False,
        'gamma': 1,
        'beta': 1,
        'tlist': np.arange(0, 1, .1),
        'iters': 500
    }
    experiment = QAOA(**parameters)
    experiment.iterate()
