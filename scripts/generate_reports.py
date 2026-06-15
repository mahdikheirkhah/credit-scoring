# scripts/generate_reports.py
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

def generate_leaf_comparison_plot(model_path: str, output_path: str):
    """
    Extracts the internal Logistic Regression equations from the Piecewise model
    and plots the top risk drivers for two distinct customer segments.
    """
    try:
        logger.info(f"Loading custom model from {model_path}...")
        pipeline_data = joblib.load(model_path)
        model = pipeline_data['model']
        feature_names = pipeline_data['selected_feature_names']
        
        # Extract the equations using the method we built
        audit_dict = model.extract_leaf_coefficients(feature_names)
        
        leaves = list(audit_dict.keys())
        if len(leaves) < 2:
            logger.warning("Not enough pure leaves to compare. Run with shallower tree.")
            return
            
        # Grab the first two distinct borrower personas
        leaf_1, leaf_2 = leaves[0], leaves[1]
        
        # Get the top 10 most impactful features for Leaf 1
        df1 = audit_dict[leaf_1].head(10).copy()
        df1['Segment'] = f"Segment A (Leaf {leaf_1})"
        
        # Look up those EXACT same 10 features in Leaf 2 to compare them directly
        df2 = audit_dict[leaf_2][audit_dict[leaf_2]['Feature'].isin(df1['Feature'])].copy()
        df2['Segment'] = f"Segment B (Leaf {leaf_2})"
        
        # Combine for plotting
        plot_df = pd.concat([df1, df2])
        
        # Plotting
        plt.figure(figsize=(12, 8))
        sns.set_theme(style="whitegrid")
        
        ax = sns.barplot(
            data=plot_df, 
            y='Feature', 
            x='Coefficient', 
            hue='Segment',
            palette=['#1f77b4', '#ff7f0e']
        )
        
        plt.title("Econometric ML: Shifting Risk Drivers Across Customer Segments", fontsize=16, pad=20)
        plt.xlabel("Logistic Regression Coefficient Magnitude (Risk Impact)", fontsize=12)
        plt.ylabel("Engineered Feature", fontsize=12)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        logger.info(f"Successfully generated Segment Comparison plot at {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate leaf comparison report: {e}")
        raise

if __name__ == "__main__":
    generate_leaf_comparison_plot(
        model_path="results/model/piecewise_model.pkl", 
        output_path="results/model/leaf_coefficient_comparison.png"
    )