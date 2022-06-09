import warnings
from pathlib import Path

import numpy as np
from pymatgen.analysis.local_env import IsayevNN
from pymatgen.io.cif import CifParser

from atomic_properties import raw_features
from radial_features import get_radial_features


def generate_graph(cif_str):
    cif_path = Path(cif_str)
    assert cif_path.exists(), f"{cif_str} DOES NOT EXIST."

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cif_parser = CifParser(cif_path)
        cif_struct = cif_parser.get_structures(primitive=False)[0]

    atom_symbols = [atom.specie.symbol for atom in cif_struct]
    atom_labels = []
    label_counter = {element: 0 for element in set(atom_symbols)}
    for symbol in atom_symbols:
        label_counter[symbol] += 1
        atom_labels.append(f"{symbol}{label_counter[symbol]}")

    cif_dict = cif_parser.as_dict().popitem()[1]
    charge_dict = {
        label: float(charge)
        for label, charge in zip(
            cif_dict["_atom_site_label"], cif_dict["_atom_type_partial_charge"]
        )
    }
    atom_charges = np.array([charge_dict[label] for label in atom_labels])

    simple_features = np.array([raw_features[symbol] for symbol in atom_symbols])
    radial_features = get_radial_features(cif_struct, atom_symbols, atom_labels)

    cif_bond_info = (
        IsayevNN(tol=0.5).get_bonded_structure(structure=cif_struct).as_dict()
    )["graphs"]["adjacency"]
    bonds = np.array(
        [(i, bond["id"]) for i, bonds in enumerate(cif_bond_info) for bond in bonds]
    )

    return {
        "node_label": np.array(atom_labels),
        "node_class": np.array(atom_symbols),
        "node_target": atom_charges,
        "node_simple_feature": simple_features,
        "node_radial_feature": radial_features,
        "edges": bonds,
    }
