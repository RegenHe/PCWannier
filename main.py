# main.py

import argparse

from PCWannier.Utils import global_data
from PCWannier.Utils import wannier_tools
from PCWannier.IncarParser import IncarParser
import PCWannier.MeshData as MeshData

def parse_args():
    parser = argparse.ArgumentParser(description="PCWannier v0.1.0")
    parser.add_argument('-i', '--input', help='Incar file path', required=True)
    parser.add_argument('-t', '--threads', type=int, default=1, help='Number of threads to use')
    return parser.parse_args()

def main():
    args = parse_args()
    global_data.threads = args.threads
    print(f"Running with {args.threads} threads")
    parser = IncarParser(args.input)
    wtools = wannier_tools()
    wtools.set_incar(parser.parse_file())
    wtools.preprocess()
    print(global_data.incar)

    mesh = MeshData.load_comsol_mesh(global_data.incar.mesh_file)
    raw_data = MeshData.load_comsol_data(global_data.incar.dataset_file)

    idxs, dists = MeshData.match_data_to_mesh(mesh, raw_data)
    raw_data.value_matrix = raw_data.value_matrix[idxs]

    epsilon = MeshData.load_comsol_data(global_data.incar.dielectric_file)

    idxs, dists = MeshData.match_data_to_mesh(mesh, epsilon)
    eps = epsilon.value_matrix[idxs]

    MeshData.distribute_data(mesh, raw_data)
    global_data.state_collection.mesh.plot_mesh()
    global_data.state_collection.plot_field(0, 0, 1)


if __name__ == '__main__':
    main()
