from multiprocessing import Pool
from pathlib import Path

import numpy as np

from generate_graph import generate_graph

n_procs = 2
src_path = Path("example_input")
dst_path = Path("example_output")


def main(cif_path):
    npz_name = cif_path.name.replace(".cif", ".npz")
    np.savez(dst_path.joinpath(npz_name), **generate_graph(cif_path))


if __name__ == "__main__":
    with Pool(processes=n_procs) as pool:
        src_glob = list(src_path.glob("*.cif"))
        for _ in pool.imap_unordered(main, src_glob):
            continue
