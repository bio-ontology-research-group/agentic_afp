from src.deepgo.metrics import compute_metrics
import pandas as pd
import numpy as np
from src.ontology import Ontology
import click as ck
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics_single_row(test_df_row, go, terms_dict, terms, ont, preds_row):
    """
    Modified version of compute_metrics to work with a single row and return metrics
    You may need to adjust this based on your actual compute_metrics implementation
    """
    # This is a placeholder - you'll need to modify based on your actual compute_metrics function
    # Assuming it returns a dictionary or tuple with the metrics
    fmax, smin, threshold, _, _, auc, aupr, _, _ = compute_metrics(test_df_row, go, terms_dict, terms, ont, preds_row)
    
    return {
        'fmax': fmax,
        'smin': smin, 
        'aupr': aupr,
        'auc': auc,
        'threshold': threshold
    }

def evaluate_per_row(test_filename, onts=['mf', 'bp', 'cc']):
    """Evaluate metrics for each row individually"""
    test_df = pd.read_pickle(test_filename)
    all_results = []
    
    for ont in onts:
        print(f"Evaluating {ont} row by row...")
        go = Ontology("data/go.obo", with_rels=True)
        terms = pd.read_pickle(f"data/{ont}_terms.pkl")['terms'].values.flatten()
        terms_dict = {v: i for i, v in enumerate(terms)}
        train_data_file = f"data/train_data.pkl"
        test_data_file = f"data/test_data.pkl"
        train_annots = pd.read_pickle(train_data_file)['prop_annotations'].values
        train_annots = list(map(lambda x: set(x), train_annots))
        test_annots = test_df['prop_annotations'].values
        test_annots = list(map(set, test_annots))
        go.calculate_ic(train_annots + test_annots)
        
        preds = test_df[f"{ont}_preds"].values
        preds = np.stack(preds, axis=0)
        
        # Evaluate each row separately
        for row_idx in range(len(test_df)):
            print(f"Processing row {row_idx + 1}/{len(test_df)}")
            
            # Get single row data
            test_df_row = test_df.iloc[[row_idx]]
            preds_row = preds[[row_idx]]
            
            try:
                # Compute metrics for this row
                metrics = compute_metrics_single_row(test_df_row, go, terms_dict, 
                                                   list(terms_dict.keys()), ont, preds_row)
                
                # Store results
                result = {
                    'filename': test_filename,
                    'ontology': ont,
                    'row_id': row_idx,
                    **metrics
                }
                all_results.append(result)
                
            except Exception as e:
                print(f"Error processing row {row_idx}: {e}")
                # Store NaN values for failed rows
                result = {
                    'filename': test_filename,
                    'ontology': ont,
                    'row_id': row_idx,
                    'fmax': np.nan,
                    'smin': np.nan,
                    'aupr': np.nan,
                    'auc': np.nan,
                    'threshold': np.nan
                }
                all_results.append(result)
    
    return pd.DataFrame(all_results)

def analyze_differences(normal_results, refined_results, onts=['bp']):
    """Analyze differences between normal and refined predictions"""
    
    # Merge results on row_id and ontology
    merged = normal_results.merge(refined_results, on=['row_id', 'ontology'], 
                                 suffixes=('_normal', '_refined'))
    
    # Calculate differences (refined - normal)
    metrics = ['fmax', 'smin', 'aupr', 'auc', 'threshold']
    for metric in metrics:
        merged[f'{metric}_diff'] = merged[f'{metric}_refined'] - merged[f'{metric}_normal']
    
    # Create scatter plots for each ontology
    for ont in onts:
        ont_data = merged[merged['ontology'] == ont].copy()
        if ont_data.empty:
            continue
            
        # Sort by row_id for better visualization
        ont_data = ont_data.sort_values('row_id')
        
        # Create scatter plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                # Remove rows with NaN values for this metric
                valid_data = ont_data.dropna(subset=[f'{metric}_normal', f'{metric}_refined'])
                
                if not valid_data.empty:
                    x_pos = valid_data['row_id']
                    
                    # Plot both normal and refined values
                    axes[i].scatter(x_pos, valid_data[f'{metric}_normal'], 
                                  alpha=0.6, label='Normal', color='blue', s=20)
                    axes[i].scatter(x_pos, valid_data[f'{metric}_refined'], 
                                  alpha=0.6, label='Refined', color='red', s=20)
                    
                    axes[i].set_title(f'{metric.upper()} - {ont.upper()} Ontology')
                    axes[i].set_xlabel('Row ID')
                    axes[i].set_ylabel(f'{metric.upper()} Value')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add some styling
                    axes[i].set_xlim(-0.5, max(x_pos) + 0.5)
        
        # Remove empty subplot if needed
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(f'metrics_scatter_{ont}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create difference scatter plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                valid_data = ont_data.dropna(subset=[f'{metric}_diff'])
                
                if not valid_data.empty:
                    x_pos = valid_data['row_id']
                    y_diff = valid_data[f'{metric}_diff']
                    
                    # Color points based on improvement (positive) or degradation (negative)
                    colors = ['green' if diff > 0 else 'red' if diff < 0 else 'gray' 
                             for diff in y_diff]
                    
                    axes[i].scatter(x_pos, y_diff, alpha=0.6, c=colors, s=20)
                    axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
                    
                    axes[i].set_title(f'{metric.upper()} Differences - {ont.upper()} (Green=Improved, Red=Degraded)')
                    axes[i].set_xlabel('Row ID')
                    axes[i].set_ylabel(f'{metric.upper()} Difference (Refined - Normal)')
                    axes[i].grid(True, alpha=0.3)
                    
                    axes[i].set_xlim(-0.5, max(x_pos) + 0.5)
        
        # Remove empty subplot if needed
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(f'metrics_differences_scatter_{ont}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    for ont in onts:
        print(f"\nOntology: {ont}")
        ont_data = merged[merged['ontology'] == ont]
        
        for metric in metrics:
            diff_col = f'{metric}_diff'
            if diff_col in ont_data.columns:
                diff_values = ont_data[diff_col].dropna()
                if not diff_values.empty:
                    print(f"{metric.upper()}:")
                    print(f"  Mean difference: {diff_values.mean():.6f}")
                    print(f"  Median difference: {diff_values.median():.6f}")
                    print(f"  Std difference: {diff_values.std():.6f}")
                    print(f"  Min difference: {diff_values.min():.6f}")
                    print(f"  Max difference: {diff_values.max():.6f}")
                    print(f"  Improved rows: {(diff_values > 0).sum()}/{len(diff_values)}")
    
    return merged

def main():
    onts = ['bp']  # You can modify this list
    
    print("=== Evaluating Normal Predictions ===")
    test_filename_normal = "data/test_predictions_mlp.pkl"
    normal_results = evaluate_per_row(test_filename_normal, onts=onts)
    normal_results.to_csv('normal_predictions_per_row_metrics.csv', index=False)
    print("Normal results saved to 'normal_predictions_per_row_metrics.csv'")
    
    print("\n=== Evaluating Refined Predictions ===")
    test_filename_refined = "data/test_predictions_refined.pkl"
    refined_results = evaluate_per_row(test_filename_refined, onts=onts)
    refined_results.to_csv('refined_predictions_per_row_metrics.csv', index=False)
    print("Refined results saved to 'refined_predictions_per_row_metrics.csv'")
    
    print("\n=== Analyzing Differences ===")
    differences = analyze_differences(normal_results, refined_results, onts=onts)
    differences.to_csv('metrics_differences_per_row.csv', index=False)
    print("Differences saved to 'metrics_differences_per_row.csv'")
    
    print("\nAnalysis complete! Check the generated CSV files and plots.")

if __name__ == "__main__":
    main()

