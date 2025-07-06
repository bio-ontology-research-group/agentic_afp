import pandas as pd
import numpy as np

def compare_list_columns(df1, df2, ont, column_name='preds', tolerance=1e-9):
    """
    Compare a column containing lists of floats between two DataFrames.
    
    Parameters:
    df1, df2: pandas DataFrames to compare
    column_name: name of the column to compare (default: 'a')
    tolerance: floating point tolerance for comparison (default: 1e-9)
    
    Returns:
    bool: True if columns are equal, False otherwise
    """
    
    # Check if both DataFrames have the column
    column_name = f"{ont}_{column_name}"  # Adjust column name based on ontology
    if column_name not in df1.columns or column_name not in df2.columns:
        print(f"Column '{column_name}' not found in one or both DataFrames")
        return False
    
    # Check if DataFrames have same number of rows
    if len(df1) != len(df2):
        print(f"DataFrames have different lengths: {len(df1)} vs {len(df2)}")
        return False

    preds1 = df1[column_name].tolist()
    preds2 = df2[column_name].tolist()
    
    for i in range(len(preds1)):
        list1 = preds1[i]
        list2 = preds2[i]
                        
                            
        # Check if both are lists and have same length
        if len(list1) != len(list2):
            print(f"Row {i}: Lists have different lengths: {len(list1)} vs {len(list2)}")
            return False

        if np.allclose(list1, list2, atol=tolerance, rtol=tolerance):
            continue
        # Compare elements with tolerance
        for j in range(len(list1)):
            if not np.isclose(list1[j], list2[j], atol=tolerance, rtol=tolerance):
                print(f"Row {i}, Element {j}: {list1[j]} != {list2[j]} within tolerance {tolerance}")
                # return False
        
        # if not np.allclose(list1, list2, atol=tolerance, rtol=tolerance):
            # print(f"Row {i}: Lists are not equal within tolerance")
            # print(f"  List 1: {list1}")
            # print(f"  List 2: {list2}")
            # return False
    
    return True

# Example usage
if __name__ == "__main__":
    # Create sample DataFrames
    ont = "mf"
    df1 = pd.read_pickle('data/test_predictions_mlp.pkl')
    df2 = pd.read_pickle('data/test_predictions_refined_dumb.pkl')
        
    # Test equal DataFrames
    result = compare_list_columns(df1, df2, ont)
    print(f"DataFrames equal: {result}")
    

