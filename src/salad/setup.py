import os

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", "-J", default=1, type=int)

    args = parser.parse_args()

    setup_script = rf"""
export PYTHONPATH={os.path.dirname(__file__)}
export J={args.processes}
export OMP_NUM_THREADS=$J
export MKL_NUM_THREADS=$J 
export NUMBA_NUM_THREADS=$J
export NUMEXPR_NUM_THREADS=$J
renice -n 10 $$ # lower the priority of any commands started from this shell
"""
    
    print(setup_script)

if __name__ == "__main__":
    main()