from .xyz2mol import *
import numpy as np
import os
import pandas as pd
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import pickle
import multiprocessing as mp
from os import path
from typing import Dict

import rdkit.Chem.Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions

#https://www.kaggle.com/c/champs-scalar-coupling/discussion/93972#latest-573627

if 1: # Supress warnings from rdkit
    from rdkit import rdBase
    from rdkit import RDLogger
    rdBase.DisableLog('rdApp.error')
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR)

GRAPH_DIR = 'C:/Kaggle/Molecule/graph'

SYMBOL=['H', 'C', 'N', 'O', 'F']
BOND_TYPE = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
HYBRIDIZATION=[
    #Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    #Chem.rdchem.HybridizationType.SP3D,
    #Chem.rdchem.HybridizationType.SP3D2,
]
DISTANCE=[
    0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, np.inf
]
COUPLING_TYPE = ['1JHC','2JHC','3JHC','1JHN','2JHN','3JHN','2JHH','3JHH']


class Struct(object):

    def __init__(self, is_copy=False, **kwargs):
        self.add(is_copy, **kwargs)

    def add(self, is_copy=False, **kwargs):
        if not is_copy:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, copy.deepcopy(value))
                except Exception:
                    setattr(self, key, value)

    def from_dict(self, dict: Dict):
        for key, value in dict.items():
            setattr(self, key, value)
        return self

    def __str__(self):
        return str(self.__dict__)


def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    return one_hot


def make_graph(molecule_name, gb_structure, gb_scalar_coupling, ):
    #https://stackoverflow.com/questions/14734533/how-to-access-pandas-groupby-dataframe-by-key

    #----
    df = gb_scalar_coupling.get_group(molecule_name)
    # ['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'type',
    #        'scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso'],

    coupling = Struct(
        id = df.id.values,
        index = df[['atom_index_0', 'atom_index_1']].values,
        type = np.array([ one_hot_encoding(t,COUPLING_TYPE) for t in df.type.values ], np.uint8),
        type_backup = np.array([ COUPLING_TYPE.index(t) for t in df.type.values ], np.int32),
        value = df.scalar_coupling_constant.values,
    )


    #----
    df = gb_structure.get_group(molecule_name)
    df = df.sort_values(['atom_index'], ascending=True)
    # ['molecule_name', 'atom_index', 'atom', 'x', 'y', 'z']
    a   = df.atom.values.tolist()
    xyz = df[['x','y','z']].values
    mol = mol_from_axyz(a, xyz)


    #---
    assert( #check
       a == [ mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    )

    #---
    factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    feature = factory.GetFeaturesForMol(mol)

    ## ** node **
    #[ a.GetSymbol() for a in mol.GetAtoms() ]

    num_atom = mol.GetNumAtoms()
    symbol   = np.zeros((num_atom,len(SYMBOL)),np.uint8) #category 5
    acceptor = np.zeros((num_atom,1),np.uint8)
    donor    = np.zeros((num_atom,1),np.uint8)
    aromatic = np.zeros((num_atom,1),np.uint8)
    hybridization = np.zeros((num_atom,len(HYBRIDIZATION)),np.uint8) # 3
    num_h  = np.zeros((num_atom,1),np.float32)#real
    atomic = np.zeros((num_atom,1),np.float32)
    small_ring = np.zeros((num_atom,1),np.uint8)

    for i in range(num_atom):
        atom = mol.GetAtomWithIdx(i)
        symbol[i]        = one_hot_encoding(atom.GetSymbol(),SYMBOL)
        aromatic[i]      = atom.GetIsAromatic()
        hybridization[i] = one_hot_encoding(atom.GetHybridization(),HYBRIDIZATION)

        num_h[i]  = atom.GetTotalNumHs(includeNeighbors=True)
        atomic[i] = atom.GetAtomicNum()
        small_ring[i] = 1 if atom.IsInRingSize(3) or atom.IsInRingSize(4) else 0

    #[f.GetFamily() for f in feature]
    for t in range(0, len(feature)):
        if feature[t].GetFamily() == 'Donor':
            for i in feature[t].GetAtomIds():
                donor[i] = 1
        elif feature[t].GetFamily() == 'Acceptor':
            for i in feature[t].GetAtomIds():
                acceptor[i] = 1

    ## ** edge **
    num_edge = num_atom*num_atom - num_atom
    edge_index = np.zeros((num_edge,2), np.uint8)
    bond_type  = np.zeros((num_edge,len(BOND_TYPE)), np.uint8)#category
    distance   = np.zeros((num_edge,1),np.float32) #real

    ij=0
    for i in range(num_atom):
        for j in range(num_atom):
            if i==j: continue
            edge_index[ij] = [i,j]

            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                bond_type[ij] = one_hot_encoding(bond.GetBondType(), BOND_TYPE)

            distance[ij] = np.linalg.norm(xyz[i] - xyz[j])

            ij+=1
    ##-------------------

    graph = Struct(
        molecule_name = molecule_name,
        smiles = Chem.MolToSmiles(mol),
        node = np.concatenate(
                 [symbol, acceptor, donor, aromatic, hybridization, small_ring, ]
                +[num_h,atomic,]
            ,-1),

        edge = np.concatenate(
                 [bond_type, ]
                + [distance, ]
            ,-1),

        edge_index = edge_index,

        coupling = coupling,
    )
    return graph


def load_csv(num_record=None):

    DATA_DIR = 'C:/Kaggle/Molecule'

    #structure
    df_structure = pd.read_csv(DATA_DIR + '/structures.csv')

    #coupling
    df_train = pd.read_csv(DATA_DIR + '/train.csv', nrows=num_record)
    df_test  = pd.read_csv(DATA_DIR + '/test.csv', nrows=num_record)
    df_test['scalar_coupling_constant']=0
    df_scalar_coupling = pd.concat([df_train,df_test])

    gb_scalar_coupling = df_scalar_coupling.groupby('molecule_name')
    gb_structure       = df_structure.groupby('molecule_name')

    return gb_structure, gb_scalar_coupling


def read_pickle_from_file(pickle_file):
    with open(pickle_file,'rb') as f:
        x = pickle.load(f)
    return x


def write_pickle_to_file(pickle_file, x):
    with open(pickle_file, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)


def do_one(p):
    i, molecule_name, gb_structure, gb_scalar_coupling, graph_file = p

    if path.exists(graph_file):
        pass
    else:
        g = make_graph(molecule_name, gb_structure, gb_scalar_coupling, )
        print(i, g.molecule_name, g.smiles)
        write_pickle_to_file(graph_file, g)


##----
def run_convert_to_graph():
    graph_dir = GRAPH_DIR
    os.makedirs(graph_dir, exist_ok=True)

    gb_structure, gb_scalar_coupling = load_csv()
    molecule_names = list(gb_scalar_coupling.groups.keys())
    molecule_names = np.sort(molecule_names)

    param=[]
    for i, molecule_name in enumerate(molecule_names):

        graph_file = graph_dir + '/%s.pickle'%molecule_name
        p = (i, molecule_name, gb_structure, gb_scalar_coupling, graph_file)
        param.append(p)

    if 1:
        pool = mp.Pool(processes=16)
        pool.map(do_one, param)
    else:
        do_one(p)
        if i == 2000: exit(0)


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_convert_to_graph()#
