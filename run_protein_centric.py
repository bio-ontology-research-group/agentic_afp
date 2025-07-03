import pandas as pd
from tqdm import tqdm
from agents import CoordinatorProteinCentricAgent

import argparse
def main(data_root, ont, test):
    
    output_file = f"{data_root}/test_predictions_refined.pkl"

    coordinator = CoordinatorProteinCentricAgent(ont)
    number_of_proteins = len(coordinator.test_df)

    # Start with the initial test DataFrame
    # new_test_df = coordinator.test_df
    # new_test_df.to_pickle(output_file)
    
    if test:
        number_of_proteins = 1
        prot_id = 4
    

    for i in tqdm(range(number_of_proteins)):
        try:
            coordinator.protein_step(prot_id, verbose=True)
            new_test_df = coordinator.test_df
            new_test_df.to_pickle(output_file)
            print("__________________________________")
            coordinator.reset()
        except Exception as e:
            print(f"Error processing protein {i}: {e}")
            continue
            # break
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--ont', type=str, default='mf')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    main(args.data_root, args.ont, args.test)
    print("Done.")
