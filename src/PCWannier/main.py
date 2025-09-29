# main.py
import argparse
import multiprocessing
from PCWannier.PCWannier import PCWannier

def parse_args():
    parser = argparse.ArgumentParser(description="PCWannier v0.1.1")
    parser.add_argument('-i', '--input', help='Input file path', required=True)
    parser.add_argument('-t', '--threads', type=int, default=multiprocessing.cpu_count(), help='Number of threads to use')
    parser.add_argument('-l', '--log', default="log.txt", help='Log file')
    parser.add_argument('-b', '--base', action='store_true', help='Plot Base Functions')
    parser.add_argument('-c', '--cache', action='store_true', help='Use cache data')
    parser.add_argument('--interp', type=str, default=None, help='Path to interpolation data file')
    parser.add_argument('--interp-wannier', type=str, default=None, help='Path to interpolation wannier file')
    parser.add_argument('--interp-epsilon', type=str, default=None, help='Path to interpolation epsilon file')
    parser.add_argument('-f', '--fatband', action='store_true', help='Start fatband calculation')
    return parser.parse_args()

def main():
    args = parse_args()
    pc_wannier = PCWannier()
    pc_wannier.run(args)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
