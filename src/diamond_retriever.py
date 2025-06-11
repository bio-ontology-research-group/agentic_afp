import pandas as pd
import os
root_dir = os.path.dirname(os.path.abspath(__file__))

def get_diamond_score(sequence: str, hypothesis_function: str, data:
                      pd.DataFrame) -> float:
    """
    Retrieve the diamond score for a given sequence and hypothesis function.
    Args:
        sequence (str): The protein sequence to analyze.
        hypothesis_function (str): The GO term representing the hypothesized function.
    Returns:
        float: The diamond score for the given sequence and hypothesis function.
    """
    # idx = terms_dict[hypothesis_function]

    preds = data[data['sequences'] == sequence]['diam_preds'].values[0]

    if hypothesis_function not in preds:
        return 0.0
    else:
        return float(preds[hypothesis_function])


