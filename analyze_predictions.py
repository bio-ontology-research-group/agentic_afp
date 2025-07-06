from src.ontology import Ontology
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_problematic_predictions(test_df, go, terms_dict, terms, ont, eval_preds, 
                                  decision_threshold=0.21, 
                                  top_k_per_protein=10, min_error_magnitude=0.1,
                                  save_results=True, output_file='problematic_predictions.csv'):
    """
    Analyzes predictions row by row to identify problematic GO terms.
    
    Args:
        test_df: DataFrame with protein data and annotations
        go: GO ontology object
        terms_dict: Dictionary mapping GO terms to indices
        terms: List of GO terms
        ont: Ontology namespace
        eval_preds: Prediction matrix (proteins x GO terms)
        decision_threshold: The threshold used for classification (default: 0.21)
        top_k_per_protein: Number of top problematic terms to report per protein
        min_error_magnitude: Minimum error magnitude to consider problematic
        save_results: Whether to save results to CSV
        output_file: Output filename for results
    
    Returns:
        DataFrame with problematic predictions analysis
    """
    
    # Build ground truth labels matrix
    print("Building ground truth labels matrix...")
    labels = np.zeros((len(test_df), len(terms_dict)), dtype=np.float32)
    for i, row in enumerate(test_df.itertuples()):
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                labels[i, terms_dict[go_id]] = 1
    
    # Calculate frequency of each GO term in the dataset
    print("Calculating GO term frequencies...")
    go_term_frequencies = {}
    for go_term, term_idx in terms_dict.items():
        go_term_frequencies[go_term] = int(np.sum(labels[:, term_idx]))
    
    print(f"GO term frequencies calculated for {len(go_term_frequencies)} terms")
    
    # Lists to store problematic predictions
    problematic_predictions = []
    
    print(f"Analyzing {len(test_df)} proteins...")
    
    for protein_idx, row in enumerate(test_df.itertuples()):
        protein_id = getattr(row, 'proteins', f'protein_{protein_idx}')  # Adjust column name as needed
        
        # Get predictions and ground truth for this protein
        protein_preds = eval_preds[protein_idx]
        protein_labels = labels[protein_idx]
        
        # Calculate error magnitude for each GO term
        error_magnitudes = []
        
        for term_idx, (pred_score, true_label) in enumerate(zip(protein_preds, protein_labels)):
            go_term = terms[term_idx]
            
            # Determine prediction based on threshold
            predicted_label = 1 if pred_score >= decision_threshold else 0
            
            # Calculate different types of errors
            error_type = None
            error_magnitude = 0
            
            if true_label == 1 and predicted_label == 0:
                # False Negative: Ground truth is positive but we predicted negative
                error_type = "False Negative"
                error_magnitude = true_label - pred_score  # How much we underestimated
                
            elif true_label == 0 and predicted_label == 1:
                # False Positive: Ground truth is negative but we predicted positive
                error_type = "False Positive"
                error_magnitude = pred_score - true_label  # How much we overestimated
            
            # Only consider actual classification errors above minimum magnitude
            if error_type and error_magnitude >= min_error_magnitude:
                error_magnitudes.append({
                    'protein_idx': protein_idx,
                    'protein_id': protein_id,
                    'go_term': go_term,
                    'predicted_score': pred_score,
                    'predicted_label': predicted_label,
                    'true_label': true_label,
                    'error_type': error_type,
                    'error_magnitude': error_magnitude,
                    'term_idx': term_idx,
                    'go_term_frequency': go_term_frequencies.get(go_term, 0)
                })
        
        # Sort by error magnitude and take top K
        error_magnitudes.sort(key=lambda x: x['error_magnitude'], reverse=True)
        top_errors = error_magnitudes[:top_k_per_protein]
        
        problematic_predictions.extend(top_errors)
        
        # Print progress
        if (protein_idx + 1) % 100 == 0:
            print(f"Processed {protein_idx + 1}/{len(test_df)} proteins")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(problematic_predictions)
    
    if len(results_df) == 0:
        print("No problematic predictions found with current thresholds.")
        return pd.DataFrame()
    
    # Add GO term information if available
    if hasattr(go, 'get_term_info'):  # Adjust based on your GO object methods
        results_df['go_term_name'] = results_df['go_term'].apply(
            lambda x: getattr(go.get_term_info(x), 'name', 'Unknown') if hasattr(go, 'get_term_info') else 'Unknown'
        )
    
    # Save results
    if save_results:
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return results_df

def summarize_problematic_predictions(results_df, top_n_terms=20):
    """
    Provides summary statistics of problematic predictions.
    """
    if len(results_df) == 0:
        print("No results to summarize.")
        return
    
    print("\n" + "="*50)
    print("PROBLEMATIC PREDICTIONS SUMMARY")
    print("="*50)
    
    # Overall statistics
    print(f"Total problematic predictions: {len(results_df)}")
    print(f"Unique proteins affected: {results_df['protein_id'].nunique()}")
    print(f"Unique GO terms involved: {results_df['go_term'].nunique()}")
    
    # Error type distribution
    print("\nError Type Distribution:")
    error_counts = results_df['error_type'].value_counts()
    for error_type, count in error_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"  {error_type}: {count} ({percentage:.1f}%)")
    
    # Most problematic GO terms
    print(f"\nTop {top_n_terms} Most Problematic GO Terms:")
    term_problems = results_df.groupby('go_term').agg({
        'error_magnitude': ['count', 'mean', 'sum'],
        'error_type': lambda x: ', '.join(x.unique()),
        'go_term_frequency': 'first'  # Get the frequency (same for all rows of same GO term)
    }).round(3)
    
    term_problems.columns = ['Error_Count', 'Avg_Error_Magnitude', 'Total_Error_Magnitude', 'Error_Types', 'Dataset_Frequency']
    term_problems = term_problems.sort_values('Total_Error_Magnitude', ascending=False)
    
    print(term_problems.head(top_n_terms))
    
    # Most problematic proteins
    print(f"\nTop 10 Proteins with Most Problematic Predictions:")
    protein_problems = results_df.groupby('protein_id').agg({
        'error_magnitude': ['count', 'mean', 'sum'],
        'error_type': lambda x: ', '.join(x.unique())
    }).round(3)
    
    protein_problems.columns = ['Count', 'Avg_Error_Magnitude', 'Total_Error_Magnitude', 'Error_Types']
    protein_problems = protein_problems.sort_values('Total_Error_Magnitude', ascending=False)
    
    print(protein_problems.head(10))

def analyze_go_term_statistics(results_df, go_term_frequencies):
    """
    Provides additional analysis of GO term statistics.
    """
    if len(results_df) == 0:
        print("No results to analyze.")
        return
    
    print("\n" + "="*60)
    print("GO TERM FREQUENCY ANALYSIS")
    print("="*60)
    
    # Add frequency information to results
    results_with_freq = results_df.copy()
    
    # Analyze problematic terms by frequency bins
    freq_bins = [0, 5, 20, 50, 100, 500, float('inf')]
    freq_labels = ['Very Rare (1-5)', 'Rare (6-20)', 'Uncommon (21-50)', 
                   'Common (51-100)', 'Frequent (101-500)', 'Very Frequent (500+)']
    
    results_with_freq['frequency_bin'] = pd.cut(results_with_freq['go_term_frequency'], 
                                               bins=freq_bins, labels=freq_labels, right=True)
    
    print("Error distribution by GO term frequency:")
    freq_analysis = results_with_freq.groupby('frequency_bin').agg({
        'error_magnitude': ['count', 'mean'],
        'go_term': 'nunique'
    }).round(3)
    
    freq_analysis.columns = ['Total_Errors', 'Avg_Error_Magnitude', 'Unique_GO_Terms']
    print(freq_analysis)
    
    # Terms with high error rate relative to their frequency
    print(f"\nGO terms with highest error rate (errors per occurrence):")
    term_error_rates = results_df.groupby('go_term').agg({
        'error_magnitude': 'count',
        'go_term_frequency': 'first'
    })
    
    # Calculate error rate (avoid division by zero)
    term_error_rates['error_rate'] = term_error_rates['error_magnitude'] / np.maximum(term_error_rates['go_term_frequency'], 1)
    term_error_rates = term_error_rates.sort_values('error_rate', ascending=False)
    term_error_rates.columns = ['Error_Count', 'Dataset_Frequency', 'Error_Rate']
    
    # Only show terms with at least 2 errors and frequency > 1
    filtered_rates = term_error_rates[(term_error_rates['Error_Count'] >= 2) & 
                                     (term_error_rates['Dataset_Frequency'] > 1)]
    print(filtered_rates.head(15))
    
    return results_with_freq
    """
    Creates visualizations of the error analysis.
    """
    if len(results_df) == 0:
        print("No results to plot.")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Error magnitude distribution
    axes[0, 0].hist(results_df['error_magnitude'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution of Error Magnitudes')
    axes[0, 0].set_xlabel('Error Magnitude')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Error type distribution
    error_counts = results_df['error_type'].value_counts()
    axes[0, 1].pie(error_counts.values, labels=error_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Error Type Distribution')
    
    # 3. GO term frequency distribution for problematic terms
    axes[0, 2].hist(results_df['go_term_frequency'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Frequency Distribution of Problematic GO Terms')
    axes[0, 2].set_xlabel('GO Term Frequency in Dataset')
    axes[0, 2].set_ylabel('Number of Problematic Predictions')
    axes[0, 2].set_yscale('log')
    
    # 4. Prediction score vs True label scatter plot
    colors = {'False Negative': 'red', 
              'False Positive': 'blue'}
    
    for error_type in results_df['error_type'].unique():
        mask = results_df['error_type'] == error_type
        axes[1, 0].scatter(results_df[mask]['true_label'], 
                          results_df[mask]['predicted_score'],
                          c=colors.get(error_type, 'gray'), 
                          label=error_type, alpha=0.6)
    
    axes[1, 0].set_xlabel('True Label')
    axes[1, 0].set_ylabel('Predicted Score')
    axes[1, 0].set_title('Prediction Score vs True Label')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Top problematic GO terms
    top_terms = results_df.groupby('go_term')['error_magnitude'].sum().nlargest(15)
    axes[1, 1].barh(range(len(top_terms)), top_terms.values)
    axes[1, 1].set_yticks(range(len(top_terms)))
    axes[1, 1].set_yticklabels([term[:15] + '...' if len(term) > 15 else term 
                               for term in top_terms.index])
    axes[1, 1].set_xlabel('Total Error Magnitude')
    axes[1, 1].set_title('Top 15 Most Problematic GO Terms')
    
    # 6. Error magnitude vs GO term frequency scatter plot
    term_stats = results_df.groupby('go_term').agg({
        'error_magnitude': 'sum',
        'go_term_frequency': 'first'
    })
    
    axes[1, 2].scatter(term_stats['go_term_frequency'], term_stats['error_magnitude'], 
                      alpha=0.6, s=50)
    axes[1, 2].set_xlabel('GO Term Frequency in Dataset')
    axes[1, 2].set_ylabel('Total Error Magnitude')
    axes[1, 2].set_title('Error Magnitude vs GO Term Frequency')
    axes[1, 2].set_xscale('log')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('problematic_predictions_analysis.png', dpi=300, bbox_inches='tight')
        print("Plots saved as 'problematic_predictions_analysis.png'")
    
    plt.show()

# Example usage:
def main():
    """
    Main function to run the analysis. 
    Adjust the parameters according to your data structure.
    """
    # Assuming you have these variables from your original script:
    # test_df, go, terms_dict, terms, ont, eval_preds

    ont = 'cc'
    
    test_df = pd.read_pickle('data/test_predictions_mlp.pkl')  # Load your test DataFrame
    eval_preds = test_df[f'{ont}_preds'].values
    eval_preds = np.stack(eval_preds, axis=0)
    go = Ontology('data/go.obo')

    terms = pd.read_pickle(f'data/{ont}_terms.pkl').values.flatten()
    terms_dict = {term: idx for idx, term in enumerate(terms)}

    preds = test_df[f'{ont}_preds'].values
    preds = np.stack(preds, axis=0)

    # Run the analysis
    results_df = analyze_problematic_predictions(
        test_df=test_df,
        go=go,
        terms_dict=terms_dict,
        terms=terms,
        ont=ont,
        eval_preds=eval_preds,
        decision_threshold=0.41,    # Your optimal threshold: 0.21 for mf, 0.41 for cc
        top_k_per_protein=10,       # Number of top problematic terms per protein
        min_error_magnitude=0.1,    # Minimum error to consider (lowered since threshold is lower)
        save_results=True,
        output_file='problematic_predictions.csv'
    )
    
    # Generate summary
    summarize_problematic_predictions(results_df, top_n_terms=20)
    
    # Generate additional GO term frequency analysis
    results_with_freq = analyze_go_term_statistics(results_df, 
                                                  {term: results_df[results_df['go_term']==term]['go_term_frequency'].iloc[0] 
                                                   for term in results_df['go_term'].unique()})
    
    # Create visualizations
    plot_error_analysis(results_df, save_plots=True)
    
    return results_df

# Uncomment to run:
results = main()










