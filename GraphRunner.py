from mol.mol2graph import run_convert_to_graph
import os

from rdkit import rdBase
from rdkit import RDLogger
rdBase.DisableLog('rdApp.error')
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_convert_to_graph()#