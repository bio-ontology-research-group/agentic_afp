import pandas as pd
from tqdm import tqdm
from agents import CoordinatorProteinCentricAgent



def main(data_root, ont):
    
    output_file = f"{data_root}/{ont}/predictions_refined.pkl"

    
    
    coordinator = CoordinatorProteinCentricAgent(ont)
    number_of_proteins = len(coordinator.test_df)
    test_protein = 0
    
    for i in tqdm(range(number_of_proteins)):
        coordinator.protein_step(test_protein, verbose=True)
        new_test_df = coordinator.test_df
        new_test_df.to_pickle(output_file)
        print("__________________________________")
        coordinator.reset()
        break
    
if __name__ == "__main__":
    data_root = 'data'
    ont = 'mf'  # or 'bp', 'cc'
    main(data_root, ont)
    print("Done.")
