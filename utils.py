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

import numpy as np

def calc_rho(reslen, contact_map, cid_list):
    rho = np.zeros(reslen)
    sep = 12
    for ia in range(reslen):
        i_chno = cid_list[ia][0]
        for ja in range(reslen):
            j_chno = cid_list[ja][0]
            if i_chno == j_chno:
                if ja >= ia - sep and ja <= ia + sep:
                    continue
                if contact_map[ia, ja] <= 10.0:
                    theta = calc_theta(contact_map[ia, ja], 1)
                    rho[ia] += theta
            else:
                if contact_map[ia, ja] <= 10.0:
                    theta = calc_theta(contact_map[ia, ja], 1)
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

with open('water.coeff', 'r') as f:
    k_water = float(f.readline().strip())
    water_kappa, water_kappa_sigma = map(float, f.readline().strip().split())
    threshold = float(f.readline().strip())
    contact_cutoff = float(f.readline().strip())
    n_wells = int(f.readline().strip())

    well_r_min = [0] * n_wells
    well_r_max = [0] * n_wells
    well_flag = [0] * n_wells
    for j in range(n_wells):
        well_r_min[j], well_r_max[j], well_flag[j] = map(float, f.readline().strip().split())

def compute_water_energy(rij, ires_type, jres_type, rho_i, rho_j):
    water_gamma_0_direct = get_water_gamma(0, ires_type, jres_type, 0)
    water_gamma_1_direct = get_water_gamma(0, ires_type, jres_type, 1)

    water_gamma_prot_mediated = get_water_gamma(1, ires_type, jres_type, 0)
    water_gamma_wat_mediated = get_water_gamma(1, ires_type, jres_type, 1)

    sigma_wat = 0.25 * (1.0 - math.tanh(water_kappa_sigma * (rho_i - threshold))) * \
                (1.0 - math.tanh(water_kappa_sigma * (rho_j - threshold)))
    sigma_prot = 1.0 - sigma_wat

    sigma_gamma_direct = (water_gamma_0_direct + water_gamma_1_direct) / 2
    sigma_gamma_mediated = sigma_prot * water_gamma_prot_mediated + sigma_wat * water_gamma_wat_mediated

    t_min_direct = math.tanh(water_kappa * (rij - well_r_min[0]))
    t_max_direct = math.tanh(water_kappa * (well_r_max[0] - rij))
    theta_direct = 0.25 * (1.0 + t_min_direct) * (1.0 + t_max_direct)

    t_min_mediated = math.tanh(water_kappa * (rij - well_r_min[1]))
    t_max_mediated = math.tanh(water_kappa * (well_r_max[1] - rij))
    theta_mediated = 0.25 * (1.0 + t_min_mediated) * (1.0 + t_max_mediated)

    water_energy = -(sigma_gamma_direct * theta_direct + sigma_gamma_mediated * theta_mediated)
    return water_energy

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

with open('burial.coeff', 'r') as f:
    k_burial = float(f.readline())
    burial_kappa = float(f.readline())
    burial_ro_min = [0.0, 0.0, 0.0]
    burial_ro_max = [0.0, 0.0, 0.0]
    burial_ro_min[0], burial_ro_max[0] = map(float, f.readline().split())
    burial_ro_min[1], burial_ro_max[1] = map(float, f.readline().split())
    burial_ro_min[2], burial_ro_max[2] = map(float, f.readline().split())

def compute_burial_energy(ires_type, rho_i):
    t = [[0.0, 0.0] for _ in range(3)]
    burial_gamma = [0.0, 0.0, 0.0]
    burial_energy = 0.0

    t[0][0] = math.tanh(burial_kappa * (rho_i - burial_ro_min[0]))
    t[0][1] = math.tanh(burial_kappa * (burial_ro_max[0] - rho_i))
    t[1][0] = math.tanh(burial_kappa * (rho_i - burial_ro_min[1]))
    t[1][1] = math.tanh(burial_kappa * (burial_ro_max[1] - rho_i))
    t[2][0] = math.tanh(burial_kappa * (rho_i - burial_ro_min[2]))
    t[2][1] = math.tanh(burial_kappa * (burial_ro_max[2] - rho_i))

    burial_gamma[0] = get_burial_gamma(ires_type, 0)
    burial_gamma[1] = get_burial_gamma(ires_type, 1)
    burial_gamma[2] = get_burial_gamma(ires_type, 2)

    burial_energy += -0.5 * k_burial * burial_gamma[0] * (t[0][0] + t[0][1])
    burial_energy += -0.5 * k_burial * burial_gamma[1] * (t[1][0] + t[1][1])
    burial_energy += -0.5 * k_burial * burial_gamma[2] * (t[2][0] + t[2][1])

    return burial_energy

with open('huckel.coeff', 'r') as f:
    k_PlusPlus, k_MinusMinus, k_PlusMinus = map(float, f.readline().split())
    k_screening = float(f.readline())
    screening_length = float(f.readline())
    debye_huckel_min_sep = int(f.readline())

def compute_electrostatic_energy(rij, i_resno, j_resno, i_resname, j_resname):
    if abs(i_resno - j_resno) < debye_huckel_min_sep:
        return 0.0
    charge_i = 0.0
    charge_j = 0.0
    term_qq_by_r = 0.0
    # Check if ires_type is D, E, R, or K; if not, skip; if so, assign charge type
    if i_resname == 'R' or i_resname == 'K':
        charge_i = 1.0
    elif i_resname == 'D' or i_resname == 'E':
        charge_i = -1.0
    else:
        return 0.0
    # Check if jres_type is D, E, R, or K; if not, skip; if so, assign charge type
    if j_resname == 'R' or j_resname == 'K':
        charge_j = 1.0
    elif j_resname == 'D' or j_resname == 'E':
        charge_j = -1.0
    else:
        return 0.0
    if charge_i > 0.0 and charge_j > 0.0:
        term_qq_by_r = k_PlusPlus * charge_i * charge_j / rij
    elif charge_i < 0.0 and charge_j < 0.0:
        term_qq_by_r = k_MinusMinus * charge_i * charge_j / rij
    else:
        term_qq_by_r = k_PlusMinus * charge_i * charge_j / rij

    return  term_qq_by_r * math.exp(-k_screening * rij / screening_length)


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