# main.py

import argparse

from PCWannier.Utils import global_data
from PCWannier.Utils import WannierTools
from PCWannier.Timer import Timer, timer
from PCWannier.IncarParser import IncarParser
import PCWannier.MeshData as MeshData
import PCWannier.MSet as MSet
import PCWannier.StateInitializer as StateInitializer

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
    wtools = WannierTools()
    wtools.set_incar(parser.parse_file())
    wtools.preprocess()
    print(global_data.incar)

    mesh = MeshData.load_comsol_mesh(global_data.incar.mesh_file)
    # mesh.plot_mesh()
    raw_data = MeshData.load_comsol_data(global_data.incar.dataset_file)

    idxs, dists = MeshData.match_data_to_mesh(mesh, raw_data)
    raw_data.value_matrix = raw_data.value_matrix[idxs]

    epsilon = MeshData.load_comsol_data(global_data.incar.dielectric_file)

    idxs, dists = MeshData.match_data_to_mesh(mesh, epsilon)

    MeshData.distribute_data(mesh, raw_data)
    global_data.state_collection.epsilon = epsilon.value_matrix[idxs].flatten()

    global_data.state_collection.normalize()

    global_data.state_collection.turn_to_Bloch()
    global_data.state_collection.extention([4, 4])

    global_data.m_set = MSet.MSet()
    global_data.m_set.init_M0(global_data.state_collection)

    global_data.state_initializer = StateInitializer.StateInitializer()
    global_data.state_initializer.projection()
    # global_data.state_collection.plot_extention_epsilon()
    # global_data.state_initializer.iter(1e-6, 100)

if __name__ == '__main__':
    main()
