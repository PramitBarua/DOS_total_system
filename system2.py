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
import os
import time
import datetime
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt 
from matplotlib import cm


import NEGFGlobal
import GenerateHamiltonian
import Alpha

if __name__ == '__main__':
    start_time = time.time()
    date = datetime.datetime.now()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("Folder_Name",
                        help="address of the folder that contains 'main_control_parameter.yml' file")
    args = parser.parse_args()
 
    location = args.Folder_Name
     
    message = ['=== System 2 has Started at ' +str(date) +' ===']
    NEGFGlobal.global_write(location, 'output.out', message=message)
     
    input_parameter = NEGFGlobal.yaml_file_loader(args.Folder_Name,
                                                  'parameter.yml')
 
    a = 1.42
    ax = np.sqrt(3)*a
    bz = 3*a
    eta = 0.001
    num_orbital = 4
    NK = int(input_parameter['Number_of_k_points'])
    NE = int(input_parameter['Number_of_E_points'])
 
#     E = np.linspace(-10 + Fermi_energy*27.2114, 10 +Fermi_energy*27.2114, NE)
    E = np.linspace(int(input_parameter['E_start']), int(input_parameter['E_end']), NE)
    E = E + 1j*eta
    kz = np.linspace(-np.pi/bz, np.pi/bz, NK)
    Fermi_energy = float(input_parameter['System2']['Fermi_energy'])
 
    start_time = time.time()
 
    file_name = 'coordinate_' + input_parameter['System2']['System_name'] + '.ao'
 
    ks_matrix, overlap_matrix = NEGFGlobal.ao_file_loader(location, file_name)

#     NEGFGlobal.global_write(location, 'ks_matrix_v1.dat', num_data=ks_matrix)
#     NEGFGlobal.global_write(location, 'overlap_matrix_v1.dat', num_data=overlap_matrix)
 
    file_name = 'map_' + input_parameter['System2']['System_name'] + '.csv'
    map_file = NEGFGlobal.csv_file_loader(location, file_name)
 
    file_name = 'map_coordinate_' + input_parameter['System2']['System_name'] + '.csv'
    map_coordinate_file = NEGFGlobal.csv_file_loader(location, file_name)
 
 
#     Gr_LM = [int(item) for item in input_parameter['Num_unit_cell']['Left_part']['Middle_part'].split(', ')]
#     Gr_LT = [int(item) for item in input_parameter['Num_unit_cell']['Left_part']['Top_part'].split(', ')]
#     Gr_LB = [int(item) for item in input_parameter['Num_unit_cell']['Left_part']['Bottom_part'].split(', ')]
    tube_L = [int(item) for item in input_parameter['System2']['Left_part'].split(', ')]
 
#     Gr_RM = [int(item) for item in input_parameter['Num_unit_cell']['Right_part']['Middle_part'].split(', ')]
#     Gr_RT = [int(item) for item in input_parameter['Num_unit_cell']['Right_part']['Top_part'].split(', ')]
#     Gr_RB = [int(item) for item in input_parameter['Num_unit_cell']['Right_part']['Bottom_part'].split(', ')]
    tube_R = [int(item) for item in input_parameter['System2']['Right_part'].split(', ')]
 
    tube_M = [int(item) for item in input_parameter['System2']['Middle_part'].split(', ')]
 
    total_tube = tube_L + tube_M + tube_R
 
    atom_num = len(map_file[tube_M[0]])*num_orbital
    matrix_element = atom_num*len(total_tube)
 
    # epsilon
    target_unit_cell = [np.array([item, item]) for item in total_tube]
    H_epsilon = np.zeros((len(target_unit_cell), atom_num, atom_num), dtype=complex)
    S_epsilon = np.zeros((len(target_unit_cell), atom_num, atom_num), dtype=complex)
 
    for id, item in enumerate(target_unit_cell):
        H_epsilon[id] = GenerateHamiltonian.matrix_block(item, map_file, ks_matrix)
        S_epsilon[id] = GenerateHamiltonian.matrix_block(item, map_file, overlap_matrix)

    # hopping
    target_unit_cell = [np.array(total_tube[n:n+2]) for n in range(len(total_tube)) if n < len(total_tube)-1]
    H_hopping = np.zeros((len(target_unit_cell), atom_num, atom_num), dtype=complex)
    S_hopping = np.zeros((len(target_unit_cell), atom_num, atom_num), dtype=complex)

#     ks_matrix_v2 = np.loadtxt(os.path.join(location, 'ks_matrix_v2.dat'))
#     atom_num = len(map_file[tube_M[0]])*num_orbital
#     matrix_element = atom_num*len(total_tube)
#     H_v2 = ks_matrix_v2[:matrix_element, :matrix_element]
#  
#     H_v1 = (Alpha.block_diagonal(H_epsilon, 0)
#             + Alpha.block_diagonal(H_hopping, 1)
#             + np.matrix.getH(Alpha.block_diagonal(H_hopping, 1)))

    pkl_filename = os.path.join(location, 'Sigmas.pkl')
    file2 = open(pkl_filename, 'rb')
    sigma00, sigma01 = pickle.load(file2)
    file2.close()
 
    for id, item in enumerate(target_unit_cell):
        H_hopping[id] = GenerateHamiltonian.matrix_block(item, map_file, ks_matrix)
        S_hopping[id] = GenerateHamiltonian.matrix_block(item, map_file, overlap_matrix)
 
    id_tube_L = np.where(np.in1d(np.array(total_tube), np.array(tube_L)))[0]
    id_tube_R = np.where(np.in1d(np.array(total_tube), np.array(tube_R)))[0]
 
#     DOS = np.zeros(len(E), dtype=complex)
#     DOS_dis = np.zeros((len(E), len(H_epsilon)), dtype=complex)
#     for idE, item_E in enumerate(E):
# #         print(idE)
#         H_epsilon_buf = np.copy(H_epsilon)
#         H_hopping_buf = np.copy(H_hopping)
#      
# #         for SSitem in id_tube_L:
# #             if SSitem == id_tube_L[-1]:
# #                 H_epsilon_buf[SSitem] += sigma00[idE]
# #             else:
# #                 H_epsilon_buf[SSitem] += sigma00[idE]
# #                 H_hopping_buf[SSitem] += sigma01[idE]
# #      
# #         for SSitem in id_tube_R:
# #             if SSitem == id_tube_R[-1]:
# #                 H_epsilon_buf[SSitem] += sigma00[idE]
# #             else:
# #                 H_epsilon_buf[SSitem] += sigma00[idE]
# #                 H_hopping_buf[SSitem] += sigma01[idE]
#      
#         gl = Alpha.sancho(item_E, H_epsilon[0], H_hopping[0], S_epsilon[0], S_hopping[0])
#         sigma_l = (item_E*S_hopping[0] - H_hopping[0]) @ gl @ np.matrix.getH(item_E*S_hopping[0] - H_hopping[0])
#      
#         H_epsilon_buf[0] += sigma_l
#      
#         gr = Alpha.sancho(item_E, H_epsilon[-1], H_hopping[-1], S_epsilon[-1], S_hopping[-1])
#         sigma_r = (item_E*S_hopping[-1] - H_hopping[-1]) @ gr @ np.matrix.getH(item_E*S_hopping[-1] - H_hopping[-1])
#      
#         H_epsilon_buf[-1] += sigma_r
#      
#         Heff = (Alpha.block_diagonal(H_epsilon_buf, 0)
#                 + Alpha.block_diagonal(H_hopping_buf, 1)
#                 + np.matrix.getH(Alpha.block_diagonal(H_hopping_buf, 1)))
#         S = (Alpha.block_diagonal(S_epsilon, 0)
#              + Alpha.block_diagonal(S_hopping, 1)
#              + np.matrix.getH(Alpha.block_diagonal(S_hopping, 1)))
#         Gr = np.linalg.inv((item_E*S - Heff))
#         Gr_diag = np.diag(Gr).reshape(len(H_epsilon_buf), atom_num)
#         DOS[idE] = np.trace(Gr)
#         DOS_dis[idE][:] = np.sum(Gr_diag, axis=1)
 
#     pkl_filename = os.path.join(location, 'DOS_dis.pkl')
#     afile = open(pkl_filename, 'wb')
#     pickle.dump([DOS, DOS_dis], afile)
#     afile.close()

    pkl_filename = os.path.join(location, 'DOS_dis.pkl')
    file2 = open(pkl_filename, 'rb')
    DOS, DOS_dis = pickle.load(file2)
    file2.close()

    E = E - Fermi_energy*27.2114
    E = np.real(E)

    f1, ax1 = plt.subplots(1, 1, sharex=True, sharey=True)
    ax1.plot(E, (-1/np.pi)*(np.imag(DOS)))
#     ax1.set_xlim([-2, 2])
#     ax1.set_ylim([-0.5, 20])
    ax1.set_xlabel('Energy')
    ax1.set_ylabel('DOS')
    plt.grid()
#     f.savefig(os.path.join(location, 'DOS.png'))

#     condition = (np.real(E)<2) & (np.real(E)>-2)
#     E_truncated = E[condition]
#     DOS_dis_truncated = DOS_dis[condition]
    data = (-1/np.pi)*np.imag(DOS_dis)
    f2, ax2 = plt.subplots(1, len(H_epsilon), sharex=True, sharey=True)
    for id in range(len(H_epsilon)):
        ax2[id].plot(data[:,id], E)
#         ax2[id].set_xlabel('DOS')
#         ax2[id].set_ylabel('Energy')
        ax2[id].set_xlim([0, 5])
        ax2[id].set_ylim([-2, 2])
        ax2[id].set_title('SS ' + str(id))
        ax2[id].grid()
    for ax in ax2.flat:
        ax.set(xlabel='DOS', ylabel='Energy')
        ax.label_outer()

    plt.figure()
#     dis = np.linspace(0, len(H_epsilon)+1, num=len(H_epsilon)+1)
#     plt.pcolor(data, cmap='jet', vmin=0, vmax=3)
#     plt.xlim([-2, 2])
#     plt.ylim([-2, 2])
#     plt.axis([dis.min(), dis.max(), E.min(), E.max()])
#     plt.colorbar()
    plt.imshow(data, cmap='jet', origin = 'lower', extent=(0, 11, E.min(), E.max()), vmin=0, vmax=3)
    plt.ylim([-2,2])
    plt.xlabel('Distance SS')
    plt.ylabel('Energy')
    plt.colorbar()
    plt.show()
