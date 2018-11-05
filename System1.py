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

    message = ['=== System 1 has Started at '+str(date)+ ' ===']
    NEGFGlobal.global_write(location, 'output.out', message=message)

    input_parameter = NEGFGlobal.yaml_file_loader(args.Folder_Name,
                                                  'parameter.yml')

    num_orbital = 4
    a = 1.42
    ax = np.sqrt(3)*a
    bz = 3*a
    eta = 0.001
    NK = int(input_parameter['Number_of_k_points'])
    NE = int(input_parameter['Number_of_E_points'])

#     E = np.linspace(-10 + Fermi_energy*27.2114, 10 +Fermi_energy*27.2114, NE)
    E = np.linspace(int(input_parameter['E_start']), int(input_parameter['E_end']), NE)
    E = E + 1j*eta
    kz = np.linspace(-np.pi/bz, np.pi/bz, NK)

    Nearest_atom = int(input_parameter['System1']['Nearest_atom'])

    file_name = 'coordinate_' + input_parameter['System1']['System_name'] + '.ao'

    ks_matrix, overlap_matrix = NEGFGlobal.ao_file_loader(location, file_name)

#     NEGFGlobal.global_write(location, 'ks_matrix.dat', num_data=ks_matrix)
#     NEGFGlobal.global_write(location, 'overlap_matrix.dat', num_data=overlap_matrix)

    file_name = 'map_' + input_parameter['System1']['System_name'] + '.csv'
    map_file = NEGFGlobal.csv_file_loader(location, file_name)

    file_name = 'map_coordinate_' + input_parameter['System1']['System_name'] + '.csv'
    map_coordinate_file = NEGFGlobal.csv_file_loader(location, file_name)

    Fermi_energy = float(input_parameter['System1']['Fermi_energy'])
#     center_ss = input_parameter['Center_graphene']

    num_unit_cell_top = [int(item) for item in input_parameter['System1']['Top_part'].split(', ')]
    num_unit_cell_middle = [int(item) for item in input_parameter['System1']['Middle_part'].split(', ')]
    num_unit_cell_bottom = [int(item) for item in input_parameter['System1']['Bottom_part'].split(', ')]
    Central_Gr = input_parameter['System1']['Central_Gr']
    Central_tube = input_parameter['System1']['Central_tube']

    # top part
    hamiltonian_block = []
    overlap_block = []
    distance = np.zeros((3,3), dtype = float)
    for id, item in enumerate(num_unit_cell_top):
        num_unit_cell = np.array([Central_Gr, item], dtype=int)
        distance[id] = Alpha.distance(num_unit_cell, map_coordinate_file)
        hamiltonian = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, ks_matrix)
        hamiltonian_block.append(hamiltonian)
 
        overlap = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, overlap_matrix)
        overlap_block.append(overlap)
 
    Hsr = Alpha.alpha_cal(np.array(hamiltonian_block), kz, distance)
    Ssr = Alpha.alpha_cal(np.array(overlap_block), kz, distance)

    # middle part
    hamiltonian_block = []
    overlap_block = []
    distance = np.zeros((3,3), dtype = float)
    for id, item in enumerate(num_unit_cell_middle):
        num_unit_cell = np.array([Central_Gr, item], dtype=int)
        distance[id] = Alpha.distance(num_unit_cell, map_coordinate_file)
        hamiltonian = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, ks_matrix)
        hamiltonian_block.append(hamiltonian)

        overlap = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, overlap_matrix)
        overlap_block.append(overlap)

    Hs = Alpha.alpha_cal(np.array(hamiltonian_block), kz, distance)
    Ss = Alpha.alpha_cal(np.array(overlap_block), kz, distance)

    # bottom part 
    hamiltonian_block = []
    overlap_block = []
    distance = np.zeros((3,3), dtype = float)
    for id, item in enumerate(num_unit_cell_bottom):
        num_unit_cell = np.array([Central_Gr, item], dtype=int)
        distance[id] = Alpha.distance(num_unit_cell, map_coordinate_file)
        hamiltonian = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, ks_matrix)
        hamiltonian_block.append(hamiltonian)
 
        overlap = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, overlap_matrix)
        overlap_block.append(overlap)
 
    Hls = Alpha.alpha_cal(np.array(hamiltonian_block), kz, distance)
    Sls = Alpha.alpha_cal(np.array(overlap_block), kz, distance)

    # graphene tube
    hamiltonian_block = []
    overlap_block = []
    distance = np.zeros((1,3), dtype = float)
    num_unit_cell = np.array([Central_tube, Central_Gr], dtype=int)
    hamiltonian = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, ks_matrix)
    hamiltonian_block.append(hamiltonian)

    overlap = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, overlap_matrix)
    overlap_block.append(overlap)

    H_beta = Alpha.alpha_cal(np.array(hamiltonian_block), kz, distance)
    S_beta = Alpha.alpha_cal(np.array(overlap_block), kz, distance)

    #graphene graphene
    hamiltonian_block = []
    overlap_block = []
    distance = np.zeros((1,3), dtype = float)

    num_unit_cell = np.array([Central_Gr, Central_Gr], dtype=int)
    hamiltonian = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, ks_matrix)
    hamiltonian_block.append(hamiltonian)

    overlap = GenerateHamiltonian.matrix_block(num_unit_cell, map_file, overlap_matrix)
    overlap_block.append(overlap)
    # this is not needed np.array(hamiltonian_block) == H_graphene. same for S_graphene
    H_graphene = Alpha.alpha_cal(np.array(hamiltonian_block), kz, distance)
    S_graphene = Alpha.alpha_cal(np.array(overlap_block), kz, distance)



    tube_shape = H_beta.shape[1]
    sigma00 = np.zeros((NE, tube_shape, tube_shape), dtype=complex)
    sigma01 = np.zeros((NE, tube_shape, tube_shape), dtype=complex)
    px_00 = np.zeros(NE, dtype=complex)
    py_00 = np.zeros(NE, dtype=complex)
    px_01 = np.zeros(NE, dtype=complex)
    py_01 = np.zeros(NE, dtype=complex)
    for idE, energy_item in enumerate(E):
        print(idE)
        message = ['=== Energy index = ' + str(idE) + ' ===']
        NEGFGlobal.global_write(location, 'output.out', message=message)

        Heff_00 = 0
        Heff_01 = 0
        for idk, item_k in enumerate(kz):
            gl = Alpha.sancho(energy_item, Hs[idk], Hls[idk], Ss[idk], Sls[idk])
            sigma_l = (energy_item*Sls[idk] - Hls[idk]) @ gl @ np.matrix.getH(energy_item*Sls[idk] - Hls[idk])
 
            gr = Alpha.sancho(energy_item, Hs[idk], Hsr[idk], Ss[idk], Ssr[idk])
            sigma_r = (energy_item*Ssr[idk] - Hsr[idk]) @ gr @ np.matrix.getH(energy_item*Ssr[idk] - Hsr[idk])
 
            H22 = H_graphene[idk] + sigma_l + sigma_r
            sigma_buffer = (energy_item*S_beta[idk] - H_beta[idk]) @ (np.linalg.inv(energy_item*S_graphene[idk] - H22)) @ np.matrix.getH(energy_item*S_beta[idk] - H_beta[idk])
            Heff_00 += sigma_buffer
            Heff_01 += sigma_buffer*np.exp(-1j*item_k*bz)
#             H_eff += (energy_item*S_beta[idk] - H_beta[idk]) @ (energy_item*S_graphene[idk] - H22) @ np.matrix.getH(energy_item*S_beta[idk] - H_beta[idk])
 
        sigma00[idE] = Heff_00/NK
        sigma00_diag = np.diag(sigma00[idE])
        py_00[idE] = sigma00_diag[(Nearest_atom*num_orbital)+1] #y orbital 2nd atom
        px_00[idE] = sigma00_diag[(Nearest_atom*num_orbital)+3] #x orbital 2nd atom
 
        sigma01[idE] = Heff_01/NK
        sigma01_diag = np.diag(sigma01[idE])
        py_01[idE] = sigma01_diag[(Nearest_atom*num_orbital)+1] #y orbital 2nd atom
        px_01[idE] = sigma01_diag[(Nearest_atom*num_orbital)+3] #x orbital 2nd atom
 
    pkl_filename = os.path.join(location, 'Sigmas.pkl')
    afile = open(pkl_filename, 'wb')
    pickle.dump([sigma00, sigma01], afile)
    afile.close()

#     E = E - Fermi_energy*27.2114
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].set_title('Sigma_00')
    ax[0].plot(E, np.imag(py_00), label='py_imag')
    ax[0].plot(E, np.imag(px_00), label='px_imag')
    ax[0].plot(E, np.real(py_00), label='py_real')
    ax[0].plot(E, np.real(px_00), label='px_real')
    ax[0].set_xlabel('Energy')
    ax[0].set_ylabel('sigma(E)')
    ax[0].grid()
    ax[0].legend()
 
    ax[1].set_title('Sigma_01')
    ax[1].plot(E, np.imag(py_01), label='py_imag')
    ax[1].plot(E, np.imag(px_01), label='px_imag')
    ax[1].plot(E, np.real(py_01), label='py_real')
    ax[1].plot(E, np.real(px_01), label='px_real')
    ax[1].set_xlabel('Energy')
    ax[1].set_ylabel('sigma(E)')
    ax[1].grid()
    ax[1].legend()
#     f.savefig(os.path.join(location, 'Sigma.png'))
    plt.show()

    message = ['=== Time took to calculate sigmas: ' +str(time.time()-start_time) + ' ===']
    NEGFGlobal.global_write(location, 'output.out', message=message)
#     sigma11_FE = np.zeros((tube_shape[1], tube_shape[2]), dtype=complex)
#     energy_item = -0.07894735985743*27.2114 +1j*eta - 0.7 
#     sigma_00 = 0
#     sigma_01 = 0
#     for idk, item_k in enumerate(kz):
#         gl = Alpha.self_energy(energy_item, Hs[idk], Hls[idk], Ss[idk], Sls[idk])
#         sigma_l = (energy_item*Sls[idk] - Hls[idk]) @ gl @ np.matrix.getH(energy_item*Sls[idk] - Hls[idk])
#   
#         gr = Alpha.self_energy(energy_item, Hs[idk], Hsr[idk], Ss[idk], Ssr[idk])
#         sigma_r = (energy_item*Ssr[idk] - Hsr[idk]) @ gr @ np.matrix.getH(energy_item*Ssr[idk] - Hsr[idk])
#   
#         H22 = H_graphene[idk] + sigma_l + sigma_r
#         sigma_buffer = (energy_item*S_beta[idk] - H_beta[idk]) @ (np.linalg.inv(energy_item*S_graphene[idk] - H22)) @ np.matrix.getH(energy_item*S_beta[idk] - H_beta[idk])
#         sigma_00 += sigma_buffer
#         sigma_01 += sigma_buffer*np.exp(-1j*item_k*bz)
# #         H_eff += H_beta[idk] @ (energy_item*np.eye(H22.shape[0]) - H22) @ np.matrix.getH(H_beta[idk])
#   
#     sigma00_FE = sigma_00/NK
#     sigma00_diag_FE = np.diag(sigma00_FE)
#     
#     sigma01_FE = sigma_01/NK
#     sigma01_diag_FE = np.diag(sigma01_FE)
#   
#     atom_index = np.array([1, 13, 4, 8, 6, 10, 0, 12, 3, 15, 5, 9, 7, 11, 2, 14])
#   
#     buffer = sigma01_diag_FE[0::4]
# #     s_orbital = np.array([x for _,x in sorted(zip(atom_index,buffer))])
#     s_orbital = buffer[atom_index]
#   
#     buffer = sigma01_diag_FE[1::4]
# #     py_orbital = [x for _,x in sorted(zip(atom_index,buffer))]
#     py_orbital = buffer[atom_index]
#        
#     buffer = sigma01_diag_FE[2::4]
# #     pz_orbital = [x for _,x in sorted(zip(atom_index,buffer))]
#     pz_orbital = buffer[atom_index]
#        
#     buffer = sigma01_diag_FE[3::4]
# #     px_orbital = [x for _,x in sorted(zip(atom_index,buffer))]
#     px_orbital = buffer[atom_index]
#       
#     distance_Yaxis = ('0', '0', '0.4', '0.4', '0.4', '0.4', '1.5', '1.5', '1.5',
#                '1.5', '2.6', '2.6', '2.6', '2.6', '3.1', '3.1')
#       
#     y_pos = np.arange(len(distance_Yaxis))
#       
#     #imaginary part
#     f, ax = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
#     plt.suptitle('E = Ef - 0.7')
#  
#     ax[0, 0].bar(y_pos, np.imag(s_orbital), align='center', alpha=0.5)
#     ax[0, 0].set_title('s orbital')
#     ax[0, 0].set_ylabel('Imaginary part of Sigma')
#     plt.xticks(y_pos, distance_Yaxis)
#  
#     ax[0, 1].bar(y_pos, np.imag(py_orbital), align='center', alpha=0.5)
#     ax[0, 1].set_title('py orbital')
#     ax[0, 1].set_ylabel('Imaginary part of Sigma')
#     plt.xticks(y_pos, distance_Yaxis)
#  
#     ax[1, 0].bar(y_pos, np.imag(pz_orbital), align='center', alpha=0.5)
#     ax[1, 0].set_title('pz orbital')
#     ax[1, 0].set_ylabel('Imaginary part of Sigma')
#     plt.xticks(y_pos, distance_Yaxis)
#  
#     ax[1, 1].bar(y_pos, np.imag(px_orbital), align='center', alpha=0.5)
#     ax[1, 1].set_title('px orbital')
#     ax[1, 1].set_ylabel('Imaginary part of Sigma')
#     plt.xticks(y_pos, distance_Yaxis)
#  
#     #real part
#     f, ax = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
#     plt.suptitle('E = Ef - 0.7')
#  
#     ax[0, 0].bar(y_pos, np.real(s_orbital), align='center', alpha=0.5)
#     ax[0, 0].set_title('s orbital')
#     ax[0, 0].set_ylabel('Real part of Sigma')
#     plt.xticks(y_pos, distance_Yaxis)
#  
#     ax[0, 1].bar(y_pos, np.real(py_orbital), align='center', alpha=0.5)
#     ax[0, 1].set_title('py orbital')
#     ax[0, 1].set_ylabel('Real part of Sigma')
#     plt.xticks(y_pos, distance_Yaxis)
#  
#     ax[1, 0].bar(y_pos, np.real(pz_orbital), align='center', alpha=0.5)
#     ax[1, 0].set_title('pz orbital')
#     ax[1, 0].set_ylabel('Real part of Sigma')
#     plt.xticks(y_pos, distance_Yaxis)
#  
#     ax[1, 1].bar(y_pos, np.real(px_orbital), align='center', alpha=0.5)
#     ax[1, 1].set_title('px orbital')
#     ax[1, 1].set_ylabel('Real part of Sigma')
#     plt.xticks(y_pos, distance_Yaxis)

#     plt.show()
