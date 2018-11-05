#!/usr/bin/env python3

__author__ = "Pramit Barua"
__copyright__ = "Copyright 2018, INT, KIT"
__credits__ = ["Pramit Barua"]
__license__ = "INT, KIT"
__version__ = "1"
__maintainer__ = "Pramit Barua"
__email__ = ["pramit.barua@student.kit.edu", "pramit.barua@gmail.com"]

'''

'''

import numpy as np


def distance(center_ss, coordinate):
    central_SS_coordinate = coordinate[center_ss[0]]
    target_SS_coordinate = coordinate[center_ss[1]]
    return target_SS_coordinate - central_SS_coordinate


def alpha_cal(hamiltonian, kz, distance):
#     len_kx = len(kx)
    len_kz = len(kz)
    H_shape = hamiltonian[0].shape

    result = np.zeros((len_kz, H_shape[0], H_shape[1]), dtype=complex)

    for idxKz, itemKz in enumerate(kz):
        for idxH, itemH in enumerate(hamiltonian):
            kr = itemKz*distance[idxH][2]
            result[idxKz] += itemH*np.exp(1j*kr)

    return result


def greens_fun(energy, hamiltonian, overlap):
    result_shape = energy.shape + hamiltonian.shape

    result = np.zeros(result_shape, dtype=complex)

    for idE, energy_item in enumerate(energy):
        result[idE] = np.linalg.inv((energy_item*overlap) - hamiltonian)

    return result

#     H_shape = hamiltonian.shape
#     result_shape = energy.shape + (H_shape[2], H_shape[3])
# 
#     result = np.zeros(result_shape, dtype=complex)
#
#     for idE, energy_item in enumerate(energy):
#         value_inv = np.zeros(H_shape, dtype=complex)
#         for idxKx in range(H_shape[0]):
#             for idxKz in range(H_shape[1]):
#                 value = (energy_item*overlap[idxKx][idxKz]) - hamiltonian[idxKx][idxKz]
#                 value_inv[idxKx][idxKz] = np.linalg.inv(value)
#         sum_value = sum_cal(value_inv)
#         result[idE] = sum_value/(H_shape[0]*H_shape[1])
#     return result


def sum_cal(array):
    shape_array = array.shape
    value = 0

    for idx in range(shape_array[0]):
        for idy in range(shape_array[1]):
            value += array[idx][idy]

    return value


def trace_cal(array):
    shape_array = array.shape
    result = np.zeros(shape_array[0], dtype=complex)
    for idx in range(len(array)):
        result[idx] = np.trace(array[idx])
    return result


def sancho(energy, h, t0_matrix, sh, st):
    es = energy*sh-h

    e = energy*sh-h
    a = energy*st-t0_matrix
    b = energy*np.matrix.getH(st) - np.matrix.getH(t0_matrix)

    while((np.linalg.norm(abs(a), ord='fro') + np.linalg.norm(abs(b), ord='fro')) > 0.001):
        g = np.linalg.inv(e)
        bga = b @ g @ a
        agb = a @ g @ b
        e = e - bga - agb
        es = es - agb

        a = -a @ g @ a
        b = -b @ g @ b

    G = np.linalg.inv(es)

    return G


def block_diagonal(x, k):
    ''' x should be a tensor-3 (#num matrices, n,n)
        k : int
        Diagonal in question. it is 0 in case of main diagonal. 
        Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal.
    '''

    shape = x.shape
    n = shape[-1]

    absk = abs(k)

    indx = np.repeat(np.arange(n),n)
    indy = np.tile(np.arange(n),n)

    indx = np.concatenate([indx + a * n for a in range(shape[0])])
    indy = np.concatenate([indy + a * n for a in range(shape[0])])

    if k < 0:
        indx += n*absk
    else:
        indy += n*absk
#     block = np.zeros((shape[0]*n,shape[0]*n))
    block = np.zeros(((shape[0]+absk)*n, (shape[0]+absk)*n), dtype=complex)
    block[(indx, indy)] = x.flatten()

    return block