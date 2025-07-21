
import numpy as np
import os
from os import listdir
from os.path import join, isfile, isdir

from pyrosetta import *
import math
import pandas as pd
import csv
import pkg_resources
from scipy.spatial import distance, distance_matrix

# extraction of tip atom
AA_to_tip = {"ALA":"CB", "CYS":"SG", "ASP":"CG", "ASN":"CG", "GLU":"CD",
                "GLN":"CD", "PHE":"CZ", "HIS":"NE2", "ILE":"CD1", "GLY":"CA",
                "LEU":"CG", "MET":"SD", "ARG":"CZ", "LYS":"NZ", "PRO":"CG",
                "VAL":"CB", "TYR":"OH", "TRP":"CH2", "SER":"OG", "THR":"OG1"}

# AAs to numbers.
# aas = "ACDEFGHIKLMNPQRSTVWY-"

AA_to_num = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
             'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
             'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20}


# AA3 to AA1.
AA3_to_AA1 = {"ALA":"A", "CYS":"C", "ASP":"D", "ASN":"N", "GLU":"E",
                 "GLN":"Q", "PHE":"F", "HIS":"H", "ILE":"I", "GLY":"G",
                 "LEU":"L", "MET":"M", "ARG":"R", "LYS":"K", "PRO":"P",
                 "VAL":"V", "TYR":"Y", "TRP":"W", "SER":"S", "THR":"T"}

aas= ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU','GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE','PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
residuemap = dict([(aas[i], i) for i in range(len(aas))])

olt = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",  "L", "K", "M", "F", "P","S", "T", "W", "Y", "V"]
aanamemap = dict([(aas[i], olt[i]) for i in range(len(aas))])

# ATYPE

atypes = {}
types = {}
ntypes = 0
script_dir = os.path.dirname(__file__)
location = pkg_resources.resource_filename(__name__, "property/aas20.txt")
with open(location, 'r') as f:
    data = csv.reader(f, delimiter=' ')
    for line in data:
        if line[1] in types:
            atypes[line[0]] = types[line[1]]
        else:
            types[line[1]] = ntypes
            atypes[line[0]] = ntypes
            ntypes += 1

# BLOSUM SCORES

location = pkg_resources.resource_filename(__name__, 'property/BLOSUM62.txt')
blosum = [i.strip().split() for i in open(location).readlines()[1:-1]]
blosummap = dict([(l[0], np.array([int(i) for i in l[1:]])/10.0) for l in blosum])


# ROSETTA ENERGIES

energy_terms = [pyrosetta.rosetta.core.scoring.ScoreType.fa_atr,\
                pyrosetta.rosetta.core.scoring.ScoreType.fa_rep,\
                pyrosetta.rosetta.core.scoring.ScoreType.fa_sol,\
                pyrosetta.rosetta.core.scoring.ScoreType.lk_ball_wtd,\
                pyrosetta.rosetta.core.scoring.ScoreType.fa_elec,\
                pyrosetta.rosetta.core.scoring.ScoreType.hbond_bb_sc,\
                pyrosetta.rosetta.core.scoring.ScoreType.hbond_sc]
energy_names = ["fa_atr", "fa_rep", "fa_sol", "lk_ball_wtd", "fa_elec", "hbond_bb_sc", "hbond_sc"]


# MEILER
location = pkg_resources.resource_filename(__name__, "property/Meiler.csv")
temp = pd.read_csv(location).values
meiler_features = dict([(t[0], t[1:]) for t in temp])

def transform(X, cutoff=4, scaling=3.0):
    X_prime = np.maximum(X, np.zeros_like(X) + cutoff) - cutoff
    return np.arcsinh(X_prime)/scaling

def seqsep(psize, normalizer=100, axis=-1):
    ret = np.ones((psize, psize))
    for i in range(psize):
        for j in range(psize):
            ret[i,j] = abs(i-j)*1.0/100-1.0
    return np.expand_dims(ret, axis)

def get_distmaps(pose, atom1="CA", atom2="CA", default="CA"):
    psize = pose.size()
    xyz1 = np.zeros((psize, 3))
    xyz2 = np.zeros((psize, 3))
    for i in range(1, psize + 1):
        r = pose.residue(i)

        if type(atom1) == str:
            if r.has(atom1):
                xyz1[i - 1, :] = np.array(r.xyz(atom1))
            else:
                xyz1[i - 1, :] = np.array(r.xyz(default))
        else:
            xyz1[i - 1, :] = np.array(r.xyz(atom1.get(r.name(), default)))

        if type(atom2) == str:
            if r.has(atom2):
                xyz2[i - 1, :] = np.array(r.xyz(atom2))
            else:
                xyz2[i - 1, :] = np.array(r.xyz(default))
        else:
            xyz2[i - 1, :] = np.array(r.xyz(atom2.get(r.name(), default)))

    return distance_matrix(xyz1, xyz2)

def getEulerOrientation(pose):
    trans_z = np.zeros((pose.size(), pose.size(), 3))
    rot_z = np.zeros((pose.size(), pose.size(), 3))
    for i in range(1, pose.size() + 1):
        for j in range(1, pose.size() + 1):
            if i == j: continue
            rt6 = pyrosetta.rosetta.core.scoring.motif.get_residue_pair_rt6(pose, i, pose, j)
            trans_z[i - 1][j - 1] = np.array([rt6[1], rt6[2], rt6[3]])
            rot_z[i - 1][j - 1] = np.array([rt6[4], rt6[5], rt6[6]])

    trans_z = np.deg2rad(trans_z)
    rot_z = np.deg2rad(rot_z)

    output = np.concatenate([trans_z, rot_z], axis=2)
    return output


