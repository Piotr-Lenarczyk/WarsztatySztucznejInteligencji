import torch
from rdkit import Chem
from rdkit.Chem import ValenceType
from typing import List, Optional, Dict

from torch_geometric.data import Data


def extract_atom_features(atom: Chem.Atom) -> List[float]:
    """Extract standard chemical features for a given RDKit atom."""
    features = []
    # One-hot encoded atomic number
    features.extend([float(atom.GetAtomicNum() == i) for i in [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]])
    features.extend([float(atom.GetDegree() == i) for i in range(7)])
    features.extend([float(atom.GetFormalCharge() == i) for i in [-2, -1, 0, 1, 2]])
    features.extend([float(atom.GetHybridization() == h) for h in [
        Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2
    ]])
    features.append(float(atom.GetValence(which=ValenceType.EXPLICIT)))
    features.append(float(atom.GetNumRadicalElectrons()))
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.append(float(atom.GetTotalNumHs()))

    # Chirality
    chiral = atom.GetChiralTag()
    features.extend([
        float(chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CW),
        float(chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    ])
    return features


def extract_bond_features(bond: Chem.Bond) -> List[float]:
    """Extract standard chemical features for a given RDKit bond."""
    bt = bond.GetBondType()
    return [
        float(bt == Chem.BondType.SINGLE),
        float(bt == Chem.BondType.DOUBLE),
        float(bt == Chem.BondType.TRIPLE),
        float(bt == Chem.BondType.AROMATIC)
    ]


def smiles_to_graph(smiles: str, y_val: float, global_desc: List[float]) -> Optional[Data]:
    """Convert a SMILES string into a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None

    x = torch.tensor([extract_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edges, attrs = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = extract_bond_features(b)
        edges.extend([[i, j], [j, i]])
        attrs.extend([f, f])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float) if attrs else torch.empty((0, len(extract_bond_features(mol.GetBonds()[0])) if mol.GetNumBonds() else 4), dtype=torch.float)
    y = torch.tensor([y_val], dtype=torch.float)
    g_desc = torch.tensor([global_desc], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, g_desc=g_desc)