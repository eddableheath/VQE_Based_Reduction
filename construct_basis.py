# Construct a basis from a (primitive) vector
# Author: Edmund Dable-Heath
"""
    Given a sampled vector from a lattice with a particular basis:

        1. Check it's primitive and make primitive if not.
        2. Construct a new basis from the primitive vector and the old basis, preserving any chosen basis vectors.
"""

import numpy as np
from math import gcd
from functools import reduce


def check_for_primitivity(vec: np.array) -> np.array:
    """
        Checks to see if the input vector is primitive, and if not returns an updated primitive vector from it.
    :param vec: input vector, int-(m, )-ndarray
    :return: primitive vector, int-(m, )-ndarray
    """
    while reduce(gcd, vec) > 1:
        vec = (vec / reduce(gcd, vec)).astype(int)
    return vec


def check_for_coprimes(vec: np.array):
    """
        Looks for pairs of numbers in the vector that are coprime, returning the location or the first one found.
    :param vec: input vector, int-(m, )-ndarray
    :return: indices of coprime pair, int-list
    """
    for i in range(vec.shape[0]):
        for j in range(i, vec.shape[0]):
            if gcd(vec[i], vec[j]) == 1:
                return [i, j]
    return None


def extended_euclid_gcd(a: int, b: int) -> list:
    """
    Returns a list `result` of size 3 where:
    Referring to the equation ax + by = gcd(a, b)
        result[0] is gcd(a, b)
        result[1] is x
        result[2] is y
    """
    s = 0; old_s = 1
    t = 1; old_t = 0
    r = b; old_r = a

    while r != 0:
        quotient = old_r//r # In Python, // operator performs integer or floored division
        # This is a pythonic way to swap numbers
        # See the same part in C++ implementation below to know more
        old_r, r = r, old_r - quotient*r
        old_s, s = s, old_s - quotient*s
        old_t, t = t, old_t - quotient*t
    return [old_r, old_s, old_t]


def construct_basis_from_primitive_vec(vec: np.array, basis: np.array, preserve_rows=None) -> np.array:
    """
        Given a vector return a primitive vector (if not already) and construct a new basis if there exists at least one
        pair of coprimes within the vector.
    :param vec: input vector, int-(m, )-ndarray
    :param basis: problem basis, int-(m, m)-ndarray
    :param preserve_rows: which basis vectors to preserve, list
    :return: new basis matrix, int-(m, m)-ndarray
    """
    unimodular = np.zeros_like(basis)
    identity = np.eye(basis.shape[0])
    prim_vec = check_for_primitivity(vec)
    coprime_pair = check_for_coprimes(prim_vec)
    if coprime_pair is not None:
        _, a_1, a_2 = extended_euclid_gcd(prim_vec[coprime_pair[0]],
                                          prim_vec[coprime_pair[1]])
        unimodular[0] = prim_vec
        unimodular[1, coprime_pair[0]] = a_2
        unimodular[1, coprime_pair[1]] = a_1
        reduced_id = np.delete(identity, coprime_pair, axis=0)
        for i in range(reduced_id.shape[0]-1, -1, -1):
            unimodular[-1-i] = reduced_id[i]
        return unimodular @ basis
    else:
        return None

# testing
if __name__ == "__main__":
    B = np.array([[1, 2, -4],
                  [-1, -4, -3],
                  [-1, -8, 6]])
    v = np.array([4, 6, 10])
    print(v)
    new_v = check_for_primitivity(v)
    print(new_v)
    coprime_pair = check_for_coprimes(new_v)
    print(coprime_pair)
    print(new_v[coprime_pair[0]], new_v[coprime_pair[1]])
    new_b = construct_basis_from_primitive_vec(v, B.T)
    print(new_b)