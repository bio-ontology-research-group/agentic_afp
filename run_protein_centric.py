import pandas as pd
from tqdm import tqdm
from agents import CoordinatorProteinCentricAgent

import argparse
def main(data_root, test):
    
    output_file = f"{data_root}/predictions_refined.pkl"

    coordinator = CoordinatorProteinCentricAgent()
    number_of_proteins = len(coordinator.test_df)
    
    if test:
        number_of_proteins = 30
    
    for i in tqdm(range(number_of_proteins)):
        coordinator.protein_step(i, verbose=True)
        new_test_df = coordinator.test_df
        new_test_df.to_pickle(output_file)
        print("__________________________________")
        coordinator.reset()
        # break
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    main(args.data_root, args.test)
    print("Done.")
