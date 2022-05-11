import numpy as np
import scipy as sp
from qutip import *
import itertools
import sys
np.set_printoptions(threshold=sys.maxsize)


class QAOA():
    def __init__(self, B, num_qubits_Q, fix_Hp=False, fix_beta=False):
        self.B = B
        self.num_qubits_Q = num_qubits_Q
        self.iter= 0
        self.step_track = []
        self.basis_track = []
        self.magnitude_track = []
        self.basis_1_track = []
        self.basis_2_track = []
        self.basis_3_track = []
        self.basis_4_track = []
        self.avg_norm_track = []
        self.fix_Hp = fix_Hp
        self.fix_beta = fix_beta

        self.step_track.append(0)
        basis_0 = self.to_vec()
        E_0 = basis_0**2
        avg_norm_0 = round(self.avg_norm(), 2)
        basis_1 = round(np.sqrt(np.sum(E_0[0])), 2)
        basis_2 = round(np.sqrt(np.sum(E_0[1])), 2)
        basis_3 = round(np.sqrt(np.sum(E_0[2])), 2)
        # basis_4 = round(np.sqrt(np.sum(E_0[3])), 2)
        #print(basis_1)
        self.basis_1_track.append(basis_1)
        self.basis_2_track.append(basis_2)
        self.basis_3_track.append(basis_3)
        # self.basis_4_track.append(basis_4)
        self.avg_norm_track.append(avg_norm_0)

    def Basis(self):
        return self.B

    def Gram(self):
        self.basis_track.append(self.B)
        #B = self.B
        if self.fix_Hp == False:
            return self.B.dag()*self.B
        elif self.fix_Hp == True:
            return self.basis_track[0].dag()*self.basis_track[0]

    def Z(self,i):
        string = [qeye(2)]*self.num_qubits_Q
        string[i] = sigmaz()
        return tensor(string)

    def Q(self):
        x = []
        for k in range(self.num_qubits_Q):
            xk = (2**(k-1))*self.Z(k)
            x.append(xk)

        I = [qeye(2)]*self.num_qubits_Q

        return sum(x) + tensor(I)/2

    def HP(self):
        x = []
        dim_G = self.Gram().shape[0]

        for i in range(dim_G):
            for j in range(dim_G):

                string_1 = [tensor([qeye(2)]*self.num_qubits_Q)]*dim_G
                string_2 = [tensor([qeye(2)]*self.num_qubits_Q)]*dim_G
                #print(string_1)

                string_1[i] = self.Q()
                string_2[j] = self.Q()

                #print(Q(num_qubits_Q))

                QiQj = tensor(string_1)*tensor(string_2)

                xx = np.real(self.Gram()[i][0][j])*QiQj
                #print(xx)
                x.append(xx)
        return sum(x)

    def X(self,i):
        string = [qeye(2)]*(self.num_qubits_Q*self.Gram().shape[0])
        string[i] = sigmax()
        return tensor(string)

    def HD(self):
        x = []

        for i in range(self.num_qubits_Q*self.Gram().shape[0]):
            xx = self.X(i)
            x.append(xx)
        return sum(x)

    def psi_0(self):

        string = [qeye(2)]*(self.num_qubits_Q*self.Gram().shape[0])
        mat = tensor(string)
        return sum(mat.eigenstates()[1]).unit()

    def HP_i(self):
        x = [self.HP()*i*i.dag() for i in self.HP().eigenstates()[1]]
        return x

    def psi(self,gamma,beta,tlist,minimum = None):
        if self.fix_beta == False:
            if minimum is None:
                prop_HP = propagator(gamma*self.HP(),tlist)
                prop_HD = propagator(beta*self.HD(),tlist)

                return prop_HD*prop_HP*self.psi_0()

            if minimum is not None:
                prop_HP = propagator(gamma*self.HP(),tlist,unitary_mode='single')
                prop_HD = propagator(beta*self.HD(),tlist,unitary_mode='single')

                return prop_HD[minimum]*prop_HP[minimum]*self.psi_0()
        elif self.fix_beta == True:
            prop_HD = propagator(beta*self.HD(),tlist)
            arg = round((.25*np.pi)/tlist[1])
            if minimum is None:
                prop_HP = propagator(gamma*self.HP(),tlist)
                return prop_HD[arg]*prop_HP*self.psi_0()

            if minimum is not None:
                prop_HP = propagator(gamma*self.HP(),tlist,unitary_mode='single')
                #prop_HD = propagator(beta*self.HD(),tlist,unitary_mode='single')
                return prop_HD[arg]*prop_HP[minimum]*self.psi_0()

    def psi_fast(self,gamma,beta,tlist):
        identity = tensor([qeye(2)]*(self.num_qubits_Q*self.Gram().shape[0]))
        prop_HD = [(identity - 1j*beta*self.HD()*t/10)**10 for t in tlist]
        prop_HP = [(identity - 1j*gamma*self.HP()*t/10)**10 for t in tlist]

        psi = []
        for i in range(len(tlist)):
            psi.append(prop_HD[i]*prop_HP[i]*self.psi_0())

        return psi

    def expect(self, gamma, beta, tlist, separate = False):
        #if separate == False:
        return expect(self.HP(), self.psi(gamma, beta, tlist))

        #elif separate == True:
        #x = []
        #for i in range(len(self.HP_i())):
        #xi = expect(self.HP_i()[i],self.psi(gamma,beta,tlist))
        #x.append(xi)
        #return x

    def expect_min(self,gamma,beta,tlist,minimum):
        #x = []
        #for i in range(len(self.HP_i())):
        x = expect(self.HP_i(),self.psi(gamma,beta,tlist,minimum = minimum))
        #x.append(xi)
        return x


    def expect_fast(self,gamma,beta,tlist,separate = False):
        if separate == False:
            return expect(self.HP(),self.psi_fast(gamma,beta,tlist))

        elif separate == True:
            x = []
            for i in range(len(self.HP_i())):
                xi = expect(self.HP_i()[i],self.psi_fast(gamma,beta,tlist))
                x.append(xi)
            return x

    def expect_low(self,gamma,beta,tlist,n):
        x = []
        for i in range(n):
            xi = expect(self.HP_i()[i],self.psi(gamma,beta,tlist))
            x.append(xi)
        return x

    def find_states(self,gamma,beta,tlist):
        exp = self.expect(gamma,beta,tlist)
        #default parameters
        minimum = np.argmin(exp)
        #print('minimum point:',minimum)
        #tlist_min =  np.arange(0,minimum*interval+interval,minimum*.0001*interval)
        exp_min = self.expect_min(.1,.1,tlist,minimum)
        #exp_min = [i[10000] for i in exp_]

        prob_min = []
        for i in range(len(exp_min)):
            if self.HP().eigenstates()[0][i] == 0:
                prob_min.append(0)
            else:
                xi = exp_min[i]/self.HP().eigenstates()[0][i]
                prob_min.append(xi)
        #prob_min[0] = 0

        E = self.HP().eigenstates()[0].tolist()
        length = len(prob_min)

        for x in range(0, 3):
            for i in range(length):
                if i <  length - 1:
                    for j in range (i+1,i+2):
                        if E[i] == E[j]:
                            prob_min[i] += prob_min[j]
                            prob_min.pop(j)
                            E.pop(j)
                            j = i
                            length-=1
                        else:
                            pass
        prob_min[0] = 1 - np.sum(prob_min)
        #print('probability:',prob_min)
        print('eigenvalue:',E)
        print('probability:',prob_min)
        return prob_min,E

    def to_vec(self):
        return np.array(self.B.data.todense().real.T)  # Return basis in colum form

    def solve(self,evalue):
        basis = self.to_vec()
        dim = basis.shape[0]
        lower = - 2**(self.num_qubits_Q)
        upper = 2**(self.num_qubits_Q) - 1

        if dim == 2:
            for a in range(lower,upper):
                for b in range(lower,upper):
                    if np.sum((a*basis[0]+b*basis[1])**2) == evalue:
                        #print('solution:',a,b,c,d)
                        v_ = np.array(a*basis[0]+b*basis[1])
                        #print('corresponding vector:',v_)
                        return v_
                    else:
                        #print('cannot be solved')
                        pass
        elif dim == 3:
            for a in range(lower,upper):
                for b in range(lower,upper):
                    for c in range(lower,upper):
                        if np.sum((a*basis[0]+b*basis[1]+c*basis[2])**2) == evalue:
                            #print('solution:',a,b)
                            v_ = np.array( a*basis[0]+b*basis[1]+c*basis[2])
                            #print('corresponding vector:',v_)
                            return v_

                        else:
                            #print('cannot be solved')
                            pass

        elif dim == 4:
            for a in range(lower,upper):
                for b in range(lower,upper):
                    for c in range(lower,upper):
                        for d in range(lower,upper):
                            if np.sum((a*basis[0]+b*basis[1]+c*basis[2]+d*basis[3])**2) == evalue:
                                #print('solution:',a,b,c,d)
                                v_ = np.array(a*basis[0]+b*basis[1]+c*basis[2]+d*basis[3])
                                #print('corresponding vector:',v_)
                                return v_

                            else:
                                v_ = None
                                pass

    def avg_norm(self):
        norm_list = []
        mat = self.to_vec()
        for i in range(mat.shape[0]):
            mag = np.sqrt(np.sum(mat[i]**2))
            norm_list.append(mag)
        return np.mean(norm_list)

    def monte_carlo(self,gamma,beta,tlist,step):
        basis = self.to_vec()
        basis_E = np.linalg.norm(basis, axis=1)

        #self.step_track.append(0)
        #self.magnitude_track.append(np.sum(basis_E))

        prob, evalues = self.find_states(gamma,beta,tlist)
        det = np.linalg.det(basis)

        chosen_E = 9e10
        det_new = 0
        basis_new =  np.zeros((self.B.dims[0][0],self.B.dims[0][0]))
        rand_vec = None
        #print(rand_vec)

        while max(basis_E) <= chosen_E or abs(round(det)) != abs(round(det_new)) or (basis_new == basis).all() == True or rand_vec is None:
            self.iter +=1
            if self.iter >= step:
                print('max step achieved:',step)
                return None

            else:
                rand_ind = np.random.choice(np.arange(len(prob)), p=prob)
                chosen_E = evalues[rand_ind]
                chosen_P = prob[rand_ind]
                rand_vec = self.solve(chosen_E)


                if rand_vec is None:
                    rand_vec = [0]*self.B.dims[0][0]
                    chosen_E = 9e10
                    print('solution is None')

                possible_list = []   #append all basis vectors that can potentially be replaced by the new vector
                for i in range(len(basis_E)):
                    if basis_E[i] > chosen_E:
                        possible_list.append(basis_E[i])
                    else:
                        pass

                possible_list = sorted(possible_list, reverse=True)
                #print('possible vector list:',possible_list)

                for i in range(len(possible_list)):
                    basis_ = basis.tolist()
                    #print('basis_ = ',basis_)
                    ind = basis_E.index(possible_list[i])
                    #print('index:',ind)
                    basis_[ind] = rand_vec.tolist()
                    det_ = np.linalg.det(basis_)

                    if abs(round(det_)) == abs(round(det)):
                        basis_new = basis_
                        #print('basis_new = ',basis_new)
                        break
                    else:
                        pass


                #print(basis_new)
                #det_new = np.linalg.det(basis)

                #basis[ind] = rand_vec

                det_new  = np.linalg.det(basis_new)
                #print("basis still the same? ",(basis_new == basis).all())

        basis = np.array(basis_new)
        #print('basis =',basis)
        self.B = Qobj(np.transpose(basis).tolist())
        total_E = np.sum(basis**2)
        avg_norm = self.avg_norm()
        avg_norm = round(avg_norm,2)

        E = basis**2
        basis_1 = round(np.sqrt(np.sum(E[0])),2)
        basis_2 = round(np.sqrt(np.sum(E[1])),2)
        basis_3 = round(np.sqrt(np.sum(E[2])),2)
        basis_4 = round(np.sqrt(np.sum(E[3])),2)

        self.step_track.append(self.iter)
        self.basis_track.append(self.B)
        self.magnitude_track.append(total_E)

        self.basis_1_track.append(basis_1)
        self.basis_2_track.append(basis_2)
        self.basis_3_track.append(basis_3)
        self.basis_4_track.append(basis_4)

        self.avg_norm_track.append(avg_norm)

        print('selected state:',chosen_E)
        print('probability:',chosen_P)
        print('iteration steps:',self.step_track)
        #print('total energy steps:',self.magnitude_track)
        print('basis 1:',self.basis_1_track)
        print('basis 2:',self.basis_2_track)
        print('basis 3:',self.basis_3_track)
        print('basis 4:',self.basis_4_track)
        print('average 2-norm:',self.avg_norm_track)
        print('new basis:',self.B)

        return self.B



    def iteration(self,gamma,beta,tlist,step):
        while self.iter <= step:
            print(f'iteration step: {self.iter}')
            self.B = self.monte_carlo(gamma, beta, tlist, step)
        return self.B


if __name__ == "__main__":
    B = np.array([[1,   2,  -4,   0.],
                  [-1, -4, -3,  5],
                  [-1, -8, 6,   9],
                  [9, -2, 2,  8]])


    B1 = QAOA(Qobj(B),2).iteration(1,1,np.arange(0,1,.1),step = 500)
    print(B1)

    #the QAOA class takes two arguments: Qobj(B) is the input bad basis and 2 is the number of qubits per qudit.
    #the iteration function takes 4 arguments:
    # 1,1 are the scale factors for beta and gamma. I played around with them in the beginning of the project
    # np.arnage(0,1,.1): the algorithms searches within this range for the optimal value of gamma to minimze energy
    # expectation. Beta is set to be equal to gamma.
    # step: allowed number of iteration steps.

    # I have to confess that the parameter optmization is done numerically, so the algorithm is really slow in 4D case.
    # You can start with a basis with small entries first just to see how
    # thr algorithm runs.
