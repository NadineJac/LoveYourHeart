def compute_shap_plot_voting(pipeline, user, user2=None, background_data=None):
    """
    Compute SHAP plot for VotingClassifier with optional What-If comparison.
    
    Parameters:
    - pipeline: Trained pipeline with VotingClassifier
    - user: Original user DataFrame
    - user2: Modified user DataFrame (optional, for What-If scenario)
    - background_data: Sample of training data for KernelExplainer (e.g., 100 samples)
    """

    # Preprocess user data
    import shap
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    FONTSIZE = 7
    X_user_preprocessed = pipeline.named_steps["preprocess"].transform(user)
    
    # Get encoded feature names
    encoded_feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    
    # Create prediction function for SHAP
    def predict_proba_fn(X):
        """Returns probability of positive class"""
        return pipeline.named_steps["voting"].predict_proba(X)[:, 1]
    
    # Initialize KernelExplainer
    if background_data is None:
        # Use a small summary (e.g., kmeans with k=10 or just the user itself)
        background = X_user_preprocessed  
        print("Warning: No background data provided. Using user data as background (not ideal).")
    else:
        # Preprocess background data
        background = pipeline.named_steps["preprocess"].transform(background_data)
    
    explainer = shap.KernelExplainer(predict_proba_fn, background)
    
    # Compute SHAP values (this will be SLOW without proper background data)
    shap_values = explainer.shap_values(X_user_preprocessed, nsamples=100)  # adjust nsamples for speed/accuracy tradeoff
    
    # Aggregate SHAP values by original feature
    original_importances = {}
    
    for i, encoded_name in enumerate(encoded_feature_names):
        if '__' in encoded_name:
            original_feature = encoded_name.split('__')[1].rsplit('_', 1)[0]
        else:
            original_feature = encoded_name.split('_')[0]
        
        shap_value = shap_values[0][i]  # Note: different indexing for KernelExplainer
        
        if original_feature in original_importances:
            original_importances[original_feature] += shap_value
        else:
            original_importances[original_feature] = shap_value
    
    # Convert SHAP values to percentages (multiply by 100)
    original_importances = {k: v * 100 for k, v in original_importances.items()}
    
    # Convert to DataFrame
    importance_df = pd.DataFrame({
        'feature': list(original_importances.keys()),
        'shap_value': list(original_importances.values())
    })

    # NEW: Add user values to feature labels
    def format_feature_label(feature_name):
        """Create label showing feature name and user's value"""
        if feature_name not in user.columns:
            return feature_name
        
        value = user[feature_name].iloc[0]
        
        # Format based on value type
        if isinstance(value, (int, float)):
            if value in [0, 1]:  # Binary encoded as 0/1
                return f"{feature_name}: {'Yes' if round(value) == 1 else 'No'}"
            elif feature_name in ['MentalHealth', 'PhysicalHealth']:
                # Add "days" suffix for health day counts
                return f"{feature_name}: {value} days"
            elif feature_name == 'SleepTime':
                # Add "hours" suffix for sleep time
                return f"{feature_name}: {value} hours"
            else:
                return f"{feature_name}: {value}"
        else:  # Categorical
            return f"{feature_name}: {value}"
    
    importance_df['feature_label'] = importance_df['feature'].apply(format_feature_label)

    # If user2 is provided, compute SHAP values for modified data
    whatif_importances = None
    changed_features = []
    if user2 is not None:
        # Find which features changed
        for col in user.columns:
            if user[col].iloc[0] != user2[col].iloc[0]:
                changed_features.append(col)
        
        # Compute SHAP values for user2
        X_user2_preprocessed = pipeline.named_steps["preprocess"].transform(user2)
        shap_values2 = explainer(X_user2_preprocessed)
        
        # Aggregate SHAP values for user2
        whatif_importances = {}
        for i, encoded_name in enumerate(encoded_feature_names):
            if '__' in encoded_name:
                original_feature = encoded_name.split('__')[1].rsplit('_', 1)[0]
            else:
                original_feature = encoded_name.split('_')[0]
            
            shap_value = shap_values2.values[0, i]
            
            if original_feature in whatif_importances:
                whatif_importances[original_feature] += shap_value
            else:
                whatif_importances[original_feature] = shap_value
        
        # Convert What-If SHAP values to percentages
        whatif_importances = {k: v * 100 for k, v in whatif_importances.items()}

    # Define modifiability levels
    HIGHLY_MODIFIABLE = {
        "Smoking",
        "AlcoholDrinking",
        "PhysicalActivity",
        "BMI",
        "SleepTime",
    }

    MODERATELY_MODIFIABLE = {
        "PhysicalHealth",
        "MentalHealth",
        "GenHealth",
        "DiffWalking",
    }

    # Classify features
    def get_modifiability(feature):
        if feature in HIGHLY_MODIFIABLE:
            return 'highly'
        elif feature in MODERATELY_MODIFIABLE:
            return 'moderately'
        else:
            return 'non'

    importance_df['modifiability'] = importance_df['feature'].apply(get_modifiability)

    # Sort by absolute value
    importance_df['abs_shap'] = importance_df['shap_value'].abs()
    importance_df = importance_df.sort_values('abs_shap', ascending=True)

    # Assign colors based on direction and modifiability
    def get_color(row):
        if row['shap_value'] < 0:  # Decreases risk (green)
            if row['modifiability'] == 'highly':
                return "#0c632c"  # Dark green
            elif row['modifiability'] == 'moderately':
                return "#569C70"  # Medium green
            else:
                return "#94d1ab"  # Light green
        else:  # Increases risk (rose)
            if row['modifiability'] == 'highly':
                return '#f43f5e'  # Dark rose
            elif row['modifiability'] == 'moderately':
                return "#f78495"  # Medium rose
            else:
                return "#fcbbc3"  # Light rose

    importance_df['color'] = importance_df.apply(get_color, axis=1)

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.barh(importance_df['feature_label'], importance_df['shap_value'], 
                color=importance_df['color'], alpha=0.9)

    # Add value labels on bars with percentage format
    for i, (bar, value, modifiability) in enumerate(zip(bars, importance_df['shap_value'], 
                                                        importance_df['modifiability'])):
        label_x = value + (0.05 if value > 0 else -0.05)  # Adjusted for percentage scale
        alignment = 'left' if value > 0 else 'right'
                                            
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', 
                va='center', ha=alignment, fontsize=8)

    # Plot What-If scenario as blue dots if available
    if whatif_importances is not None and changed_features:
        # Add What-If values as blue dots for changed features
        for idx, row in importance_df.iterrows():
            feature = row['feature']
            if feature in changed_features and feature in whatif_importances:
                whatif_value = whatif_importances[feature]
                # Use the enumerated position in the sorted dataframe
                y_position = list(importance_df['feature_label']).index(row['feature_label'])
                
                # Plot blue dot
                ax.plot(whatif_value, y_position, 'o', color='#0055A4', 
                       markersize=FONTSIZE, zorder=5)
                
                # Add label for What-If value with percentage
                label_x = whatif_value
                alignment = 'left' if whatif_value > 0 else 'right'
                ax.text(label_x, y_position, f'  {whatif_value:.1f}%  ', 
                       va='center', ha=alignment, fontsize=FONTSIZE, 
                       color='#0055A4')

    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

    # Labels and title
    ax.set_xlabel('SHAP Value (Impact on Risk %)', fontsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=FONTSIZE)
    ax.tick_params(axis="x", labelsize=FONTSIZE)

    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor='#f43f5e', label='Risk ↑ (Highly Modifiable)', alpha=0.9),
        Patch(facecolor="#f78495", label='Risk ↑ (Moderately Modifiable)', alpha=0.9),
        Patch(facecolor="#fcbbc3", label='Risk ↑ (Non-modifiable)', alpha=0.9),
        Patch(facecolor="#0c632c", label='Risk ↓ (Highly Modifiable)', alpha=0.9),
        Patch(facecolor="#569C70", label='Risk ↓ (Moderately Modifiable)', alpha=0.9),
        Patch(facecolor="#94d1ab", label='Risk ↓ (Non-modifiable)', alpha=0.9),
    ]
    
    # Add What-If legend item if applicable
    if whatif_importances is not None and changed_features:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#0055A4', 
                   markersize=FONTSIZE, label='What-If Scenario', markeredgecolor='white')
        )
    
    ax.legend(handles=legend_elements, loc='best', fontsize=FONTSIZE, framealpha=0)

    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    current_xlim = ax.get_xlim()
    ax.set_xlim(left=current_xlim[0] * 1.1, right=current_xlim[1])  # Extend left by 10%
    return fig, importance_df