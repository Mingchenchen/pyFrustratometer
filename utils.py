import math
import Bio.PDB
from Bio.PDB.PDBParser import PDBParser
import os
import random
import numpy as np


def vector(p1, p2):
    return [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]]

def vabs(a):
    return math.sqrt(pow(a[0],2)+pow(a[1],2)+pow(a[2],2))

def vector_center(p1,p2):
    return [0.5*p2[0]+0.5*p1[0], 0.5*p2[1]+0.5*p1[1], 0.5*p2[2]+0.5*p1[2]]

def calc_dist_matrix(cb_atoms) :
    """Returns a matrix of C-alpha distances between two chains"""
    reslen = len(cb_atoms);
    answer = np.zeros((reslen, reslen))
    for i in range(len(cb_atoms)):
        atom1 = cb_atoms[i];
        for j in range(i, len(cb_atoms)):
            atom2 = cb_atoms[j];
            answer[i,j] = vabs(atom1 - atom2); answer[j,i] = answer[i, j]
    return answer

def get_index_water(pdbcode):
    se_map = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE"]
    se_map_b = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "M"]
    p = PDBParser(PERMISSIVE=1)
    s = p.get_structure(pdbcode, pdbcode+'.pdb')
    #chains = s[0].get_list()
    chains = s[0].get_list()
    ca_atoms = [];
    cb_atoms = [];
    cid_list = [];
    resname_list = [];
    #residue_one[atom_map[se_map.index(residue_one.get_resname())]];
    for chain in chains:
        for res in chain:
            cid_list.append(chain.id+str(res.get_id()[1]));
            #print res.get_id()[1]
            if (res.get_resname() in se_map) and (res.get_resname() == 'GLY' or res.has_id('CB')==0):
                ca_atoms.append(res['CA'].get_coord())
                resname_list.append(se_map_b[se_map.index(res.get_resname())])
            if (res.get_resname() in se_map) and res.has_id('CB'):
                ca_atoms.append(res['CB'].get_coord())
                resname_list.append(se_map_b[se_map.index(res.get_resname())])
            if 'CA' in res:
                cb_atoms.append(res['CA'].get_coord())
    dist_matrix = calc_dist_matrix(ca_atoms);
    return dist_matrix, cid_list, resname_list, cb_atoms

def get_residue_type(i_resid):
    se_map_s = [0, 0, 4, 3, 6, 13, 7, 8, 9, 0, 11, 10, 12, 2, 0, 14, 5, 1, 15, 16, 0, 19, 17, 0, 18, 0]
    #residue_type = se_map_s[i_resid]
    return se_map_s[ord(i_resid) - ord('A')]

rmin=4.5            #lower bound of direct contact
rmax1=6.5           #upper bound of direct contact
rmax2=9.5           #upper bound of mediated contact
threshold=2.6       #density threshold
water_kappa = 5.0         #steepness of tanh function, for calculating theta
water_kappa_sigma = 7.0   #steepness of tanh function, for calculating sigma
calc_theta_cutoff = 1e-3  #sum the energy contribution with theta > cutoff

def calc_theta(rij, i_well): 

	if i_well == 1:
		theta = 0.25 * (1+math.tanh(water_kappa*(rij-rmin))) * (1+math.tanh(water_kappa*(rmax1-rij)))
		#theta = 0.5 * np.tanh(kappa*(rij-rmin)) * np.tanh(kappa*(rmax1-rij)) + 0.5
	elif i_well == 2:
		theta = 0.25 * (1+math.tanh(water_kappa*(rij-rmax1))) * (1+math.tanh(water_kappa*(rmax2-rij)))
	return theta

def calc_rho_single_chain(ia, reslen, contact_map):
	rho = 0
	sep = 12; 
	for ja in range(reslen):
		if ja >= ia-sep and ja <= ia+sep :
			continue
		if contact_map[ia, ja] <= 10.0:
			theta= calc_theta(contact_map[ia, ja], 1)
			rho += theta
	return rho

def calc_rho(reslen, contact_map):
	rho = np.zeros(reslen)
	sep = 12; 
	for ia in range(reslen):
		for ja in range(reslen):
			if ja >= ia-sep and ja <= ia+sep :
				continue
			if contact_map[ia, ja] <= 10.0:
				theta= calc_theta(contact_map[ia, ja], 1)
				rho[ia] += theta
	return rho


def calc_sigma_single_chain(ia, ja, reslen, contact_map): 
	rho_i = calc_rho_single_chain(ia, reslen, contact_map)
	rho_j = calc_rho_single_chain(ja, reslen, contact_map)
	H_i = 0.5 * ( 1 - math.tanh(water_kappa_sigma*( rho_i - threshold ) ) )
	H_j = 0.5 * ( 1 - math.tanh(water_kappa_sigma*( rho_j - threshold ) ) )
	sigma_water = H_i * H_j 
	#print(ia, ja, rho_i, rho_j)
	return sigma_water

### water part
def readgamma(gammafile):
    if not os.path.exists(gammafile):
        raise FileNotFoundError("File gamma.dat doesn't exist")
    water_gamma = np.zeros((2, 20, 20, 2))
    with open('gamma.dat', 'r') as in_wg:
        for i_well in range(2):
            for i in range(20):
                for j in range(i, 20):
                    values = list(map(float, in_wg.readline().strip().split()))
                    #print(values)
                    water_gamma[i_well][i][j] = values;
                    # Make symmetric
                    water_gamma[i_well][j][i] = water_gamma[i_well][i][j]
    return water_gamma

water_gamma = readgamma('gamma.dat');

def get_water_gamma(i_well, ires_type, jres_type, water_prot_flag):
    return water_gamma[i_well][ires_type][jres_type][water_prot_flag]

### burial part
def readburialgamma(gammafile):
    if not os.path.exists(gammafile):
        raise FileNotFoundError("File burial_gamma.dat doesn't exist")
    burial_gamma = np.zeros((20, 3))
    with open("burial_gamma.dat", "r") as in_brg:
        for i in range(20):
            line = in_brg.readline().strip()
            burial_gamma[i, :]  = list(map(float, line.split()))
    return burial_gamma

burial_gamma = readburialgamma('burial_gamma.dat');

def get_burial_gamma(ires_type, local_dens):
    return burial_gamma[ires_type][local_dens]


