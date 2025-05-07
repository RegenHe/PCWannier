# main.py
import argparse
import multiprocessing
from PCWannier.PCWannier import PCWannier

def parse_args():
    parser = argparse.ArgumentParser(description="PCWannier v0.1.0")
    parser.add_argument('-i', '--input', help='Incar file path', required=True)
    parser.add_argument('-t', '--threads', type=int, default=1, help='Number of threads to use')
    parser.add_argument('-l', '--log', default="log.txt", help='Log file')
    parser.add_argument('-b', '--base', action='store_true', help='Plot Base Functions')
    parser.add_argument('--interp', type=str, default=None, help='Path to interpolation data file')
    return parser.parse_args()

def main():
    args = parse_args()
    pc_wannier = PCWannier()
    pc_wannier.run(args)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
