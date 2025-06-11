from src.ontology import Ontology
from typing import Dict, List, Tuple

def retrieve_ancestor_scores(go_term: str, go: Ontology, terms_dict: Dict, initial_predictions: List[float]) -> List[Tuple[str, float]]:
    """
    Retrieve scores for ancestors of a given GO term.
    Args:
        go_term (str): The GO term for which to retrieve ancestor scores.
    Returns:
        List[Tuple[str, float]]: A list of tuples containing GO term identifiers and their scores.
    """
    ancestors = go.get_ancestors(go_term)
    ancestor_ids = [terms_dict[ancestor] for ancestor in ancestors if ancestor in terms_dict]
    ancestor_scores = {go_id: initial_predictions[idx] for go_id, idx in zip(ancestors, ancestor_ids)}
    return ancestor_scores
