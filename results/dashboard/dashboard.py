import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import shap
import os
from loguru import logger

from scripts.preprocess import BaselinePreprocessor
from scripts.pipline import load_and_engineer_features

# ==========================================
# 0. INITIALIZE DROPDOWN DATA & DICTIONARY
# ==========================================
SAMPLE_PATH = "results/dashboard_sample.csv"
# Pointing to the renamed file exactly as requested
DICT_PATH = "results/dashboard/feature_descriptions.txt"

try:
    if os.path.exists(SAMPLE_PATH):
        sample_df = pd.read_csv(SAMPLE_PATH, usecols=['SK_ID_CURR'])
        valid_client_ids = sorted(sample_df['SK_ID_CURR'].unique())
        CLIENT_OPTIONS = [{'label': f"Client: {cid}", 'value': cid} for cid in valid_client_ids]
    else:
        CLIENT_OPTIONS = []
except Exception as e:
    logger.warning(f"Could not load dropdown options: {e}")
    CLIENT_OPTIONS = []

def load_feature_dictionary(filepath):
    feature_dict = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                if ':' in line:
                    parts = line.split(':', 1)
                    key = parts[0].strip()
                    val = parts[1].strip()
                    feature_dict[key] = val
    else:
        logger.warning(f"Feature dictionary not found at {filepath}")
    return feature_dict

FEATURE_DICT = load_feature_dictionary(DICT_PATH)

def get_feature_description(feature_name):
    """Attempts to match the engineered feature name back to the base dictionary description."""
    if feature_name in FEATURE_DICT:
        return FEATURE_DICT[feature_name]
        
    for key, desc in FEATURE_DICT.items():
        if key in feature_name:
            if '_MAX' in feature_name: return f"Maximum recorded value of: {desc}"
            if '_MEAN' in feature_name or '_AVG' in feature_name: return f"Average recorded value of: {desc}"
            if '_SUM' in feature_name: return f"Total sum of: {desc}"
            if '_MIN' in feature_name: return f"Minimum recorded value of: {desc}"
            if 'WAS_MISSING' in feature_name: return f"Flag indicating if data was originally missing for: {desc}"
            return f"Derived from: {desc}"
            
    custom_desc = {
        'CC_UTILIZATION': "Credit Card Utilization (Balance / Limit). High value indicates maxing out cards.",
        'CC_ATM_DRAWING_RATIO': "Ratio of ATM Cash Advances to Credit Limit. High value indicates severe cash desperation.",
        'CC_IS_LATE': "Count of months the client was late on a credit card payment.",
        'PAYMENT_DELAY': "Days between actual payment and due date (Positive = Late, Negative = Early).",
        'PAYMENT_FRACTION': "Ratio of Amount Paid to Prescribed Amount (1.0 = perfect payment, <1.0 = underpayment).",
        'APPLICATION_CREDIT_RATIO': "Ratio of Amount Applied For vs Amount Granted. >1.0 means asking for more than allowed.",
        'IS_OPEN_ENDED_CREDIT': "Flag indicating presence of revolving/credit card debt reported by the bureau.",
        'BUREAU_ACTIVE_DEBT_RATIO': "Ratio of Current Debt to Current Credit Limit on active external loans.",
        'RECENT_LOAN_FLAG': "Flag indicating application for external credit within the last 180 days (credit hunger)."
    }
    
    for custom_key, desc in custom_desc.items():
         if custom_key in feature_name:
             return desc

    return "No description available for this engineered feature."

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Econometric ML Risk Dashboard"

# ==========================================
# 1. DATA & MODEL LOADING HELPERS
# ==========================================
@dash.callback(Output('dummy', 'children'), Input('dummy', 'id'))
def dummy_load(_):
    return ""

def load_pipeline(model_type: str):
    path = f"results/model/{model_type}_model.pkl"
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def get_client_data(client_id: int):
    try:
        df = pd.read_csv(SAMPLE_PATH) 
        client_row = df[df['SK_ID_CURR'] == client_id]
        if client_row.empty:
            return None, None
        
        target = client_row['TARGET'].values[0]
        X_client = client_row.drop(columns=['SK_ID_CURR', 'TARGET'])
        return X_client, target
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        return None, None

# ==========================================
# 2. UI LAYOUT
# ==========================================
app.layout = dbc.Container([
    html.Div(id='dummy'),
    dbc.Row([
        dbc.Col(html.H2("Credit Risk Auditing Dashboard", className="text-primary mt-4 mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Control Panel", className="card-title"),
                    
                    html.Label("Select Client ID:"),
                    dcc.Dropdown(
                        id="client-input",
                        options=CLIENT_OPTIONS,
                        placeholder="Search or Select a Client ID...",
                        className="mb-3"
                    ),
                    
                    html.Label("Select Model Architecture:"),
                    dcc.Dropdown(
                        id="model-selector",
                        options=[
                            {'label': 'White Box (Global Logistic Regression)', 'value': 'logistic'},
                            {'label': 'Econometric (Piecewise Model Tree)', 'value': 'piecewise'},
                            {'label': 'Black Box (LightGBM + SHAP)', 'value': 'lightgbm'}
                        ],
                        value='piecewise',
                        className="mb-3"
                    ),
                    dbc.Button("Audit Client", id="submit-btn", color="primary", className="w-100")
                ])
            ], className="shadow-sm")
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Prediction Assessment", className="card-title"),
                    html.Div(id="prediction-output", className="mt-3")
                ])
            ], className="shadow-sm mb-4"),
            
            dbc.Card([
                dbc.CardBody([
                    html.H5("Mathematical Audit Trail (Top 10 Risk Drivers)", className="card-title"),
                    html.Div(id="tree-path-output", className="text-muted mb-2"),
                    
                    # ---> THE VISUAL GUIDE FOR HOVERING <---
                    html.Div([
                        html.Span("💡 Guide: Hover your mouse over any bar to view the business definition, the client's exact actual value, and the mathematical weight applied.")
                    ], className="alert alert-info py-2 my-3 border-0"),
                    
                    dcc.Graph(id="explainability-plot")
                ])
            ], className="shadow-sm")
        ], width=8)
    ])
], fluid=True)

# ==========================================
# 3. INTERACTIVITY & MATH LOGIC
# ==========================================
@app.callback(
    [Output("prediction-output", "children"),
     Output("tree-path-output", "children"),
     Output("explainability-plot", "figure")],
    [Input("submit-btn", "n_clicks")],
    [State("client-input", "value"),
     State("model-selector", "value")],
    prevent_initial_call=True
)
def update_dashboard(n_clicks, client_id, model_type):
    if not client_id:
        return dbc.Alert("Please select a valid Client ID.", color="warning"), "", dash.no_update
        
    X_client, actual_target = get_client_data(client_id)
    if X_client is None:
        return dbc.Alert("Client ID not found in database.", color="danger"), "", dash.no_update

    pipeline = load_pipeline(model_type)
    if pipeline is None:
        return dbc.Alert(f"Model {model_type} not found. Please train it first.", color="danger"), "", dash.no_update
    
    model_obj = pipeline['model']
    
    # Apply Dynamic Features
    X_filtered = pipeline['preprocessor_obj'].filter_features(X_client, fit=False)
    X_processed_np = pipeline['preprocessor_obj'].transform(X_filtered)
    X_processed = pd.DataFrame(X_processed_np, columns=pipeline['preprocessor_obj'].feature_names)
    
    if pipeline.get('selector') is not None:
        X_selected_np = pipeline['selector'].transform(X_processed)
        X_final = pd.DataFrame(X_selected_np, columns=pipeline['selected_feature_names'])
    else:
        X_final = X_processed

    # Generate Prediction
    prob = model_obj.predict_proba(X_final)[0, 1]
    is_correct = (prob > 0.5) == actual_target
    
    status_color = "success" if is_correct else "danger"
    status_text = "CORRECT" if is_correct else "INCORRECT"
    actual_text = "Default" if actual_target == 1 else "Good Payer"
    
    pred_ui = html.Div([
        html.H3(f"Risk Score: {prob:.2%}", className="text-info"),
        html.H5(f"Actual Outcome: {actual_text}"),
        dbc.Badge(f"Model Assessment: {status_text}", color=status_color, className="p-2 fs-6")
    ])
    
    tree_path_text = ""
    feature_names = X_final.columns.tolist()
    client_values = X_final.iloc[0].values # The exact raw values for THIS specific client

    # ---> EXTRACTING LOCAL FEATURES AND WEIGHTS <---
    if model_type == 'logistic':
        weights = model_obj.coef_[0]
        impacts = weights * client_values
        plot_df = pd.DataFrame({'Feature': feature_names, 'Client Value': client_values, 'Model Weight': weights, 'Impact': impacts})
        title = "Global Equation: Feature Value × Global Coefficient"
        
    elif model_type == 'piecewise':
        leaf_id = model_obj.tree.apply(X_final)[0]
        local_model = model_obj.leaf_models.get(leaf_id, model_obj.global_fallback)
        weights = local_model.coef_[0]
        impacts = weights * client_values
        
        plot_df = pd.DataFrame({'Feature': feature_names, 'Client Value': client_values, 'Model Weight': weights, 'Impact': impacts})
        tree_path_text = html.Strong(f"Assigned to Borrower Segment: Leaf Node {leaf_id}")
        title = f"Local Econometric Equation (Leaf {leaf_id}): Feature Value × Local Coefficient"
        
    elif model_type == 'lightgbm':
        explainer = shap.TreeExplainer(model_obj)
        shap_values = explainer.shap_values(X_final)
        
        impacts = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
            
        plot_df = pd.DataFrame({'Feature': feature_names, 'Client Value': client_values, 'Model Weight': "N/A (Tree Based)", 'Impact': impacts})
        title = "Black Box Approximation: SHAP Values"

    # Format numbers so they are readable in the hover
    plot_df['Client Value'] = plot_df['Client Value'].apply(lambda x: f"{x:,.4g}" if isinstance(x, (int, float)) else x)
    plot_df['Model Weight'] = plot_df['Model Weight'].apply(lambda x: f"{x:,.4g}" if isinstance(x, (int, float)) else x)

    plot_df['Abs_Impact'] = plot_df['Impact'].abs()
    plot_df = plot_df[plot_df['Abs_Impact'] > 1e-5]
    
    plot_df = plot_df.sort_values(by='Abs_Impact', ascending=False).head(10)
    plot_df = plot_df.sort_values(by='Abs_Impact', ascending=True) 
    
    plot_df['Color'] = np.where(plot_df['Impact'] > 0, '#2ecc71', '#e74c3c')
    
    # Map the descriptions and rename the column for a cleaner tooltip UI
    plot_df['Business Description'] = plot_df['Feature'].apply(get_feature_description)
    
    # ---> CUSTOM TOOLTIP (HOVER_DATA) <---
    fig = px.bar(
        plot_df, 
        x='Impact', 
        y='Feature', 
        orientation='h',
        title=title,
        color='Color',
        color_discrete_map="identity",
        hover_data={
            "Color": False, 
            "Feature": False, 
            "Abs_Impact": False, 
            "Client Value": True,
            "Model Weight": True,
            "Impact": ':.4f', 
            "Business Description": True
        },
        height=650 
    )
    
    fig.update_yaxes(
        type='category',
        categoryorder='array',
        categoryarray=plot_df['Feature'].tolist(),
        title=None, 
        tickfont={'size': 12}
    )
    
    fig.update_xaxes(
        title='Mathematical Impact on Final Risk Score', 
        tickfont={'size': 12}
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=40),
        # Ensures the tooltip wraps nicely and uses a readable font
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial") 
    )

    return pred_ui, tree_path_text, fig

if __name__ == '__main__':
    os.makedirs("results/dashboard", exist_ok=True)
    
    if not os.path.exists(SAMPLE_PATH):
        logger.info("First run detected: Generating Dashboard Cache...")
        preprocessor = BaselinePreprocessor()
        df_merged = load_and_engineer_features(preprocessor, dataset_type="train")
        df_merged.sample(10000, random_state=42).to_csv(SAMPLE_PATH, index=False)
        logger.info("Cache generated successfully.")
        
    logger.info("Starting Dash Server...")
    app.run(debug=True, port=8050)