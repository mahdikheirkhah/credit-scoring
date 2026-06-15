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
# 0. INITIALIZE DROPDOWN DATA
# ==========================================
# We load the IDs before the app starts so the Dropdown is populated instantly.
SAMPLE_PATH = "results/dashboard_sample.csv"
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
                    # ---> THE NEW DROPDOWN <---
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
    client_values = X_final.iloc[0].values

    # ---> THE OOP FIX IS HERE <---
    if model_type == 'logistic':
        # model_obj IS the LogisticRegression instance
        weights = model_obj.coef_[0]
        impacts = weights * client_values
        plot_df = pd.DataFrame({'Feature': feature_names, 'Impact': impacts})
        title = "Global Equation: Feature Value × Global Coefficient"
        
    elif model_type == 'piecewise':
        # model_obj IS the Piecewise wrapper instance
        leaf_id = model_obj.tree.apply(X_final)[0]
        local_model = model_obj.leaf_models.get(leaf_id, model_obj.global_fallback)
        weights = local_model.coef_[0]
        impacts = weights * client_values
        
        plot_df = pd.DataFrame({'Feature': feature_names, 'Impact': impacts})
        tree_path_text = html.Strong(f"Assigned to Borrower Segment: Leaf Node {leaf_id}")
        title = f"Local Econometric Equation (Leaf {leaf_id}): Feature Value × Local Coefficient"
        
    elif model_type == 'lightgbm':
        # model_obj IS the LGBMClassifier instance
        explainer = shap.TreeExplainer(model_obj)
        shap_values = explainer.shap_values(X_final)
        
        if isinstance(shap_values, list):
            impacts = shap_values[1][0]
        else:
            impacts = shap_values[0]
            
        plot_df = pd.DataFrame({'Feature': feature_names, 'Impact': impacts})
        title = "Black Box Approximation: SHAP Values"

    plot_df['Abs_Impact'] = plot_df['Impact'].abs()
    plot_df = plot_df.sort_values(by='Abs_Impact', ascending=False).head(10)
    plot_df['Color'] = np.where(plot_df['Impact'] > 0, '#e74c3c', '#2ecc71')
    
    fig = px.bar(
        plot_df, 
        x='Impact', y='Feature', orientation='h',
        title=title, color='Color', color_discrete_map="identity"
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
# ==========================================
    # UPDATE: Filtering, Sorting, and Anti-Ghosting
    # ==========================================
    plot_df['Abs_Impact'] = plot_df['Impact'].abs()
    
    # 1. Filter out mathematical noise
    plot_df = plot_df[plot_df['Abs_Impact'] > 1e-5]
    
    # 2. Take Top 10, then sort ASCENDING so the biggest bar is at the TOP of the chart
    plot_df = plot_df.sort_values(by='Abs_Impact', ascending=False).head(10)
    plot_df = plot_df.sort_values(by='Abs_Impact', ascending=True) 
    
    # 3. Determine colors
    plot_df['Color'] = np.where(plot_df['Impact'] > 0, '#e74c3c', '#2ecc71')
    
    fig = px.bar(
        plot_df, 
        x='Impact', 
        y='Feature', 
        orientation='h',
        title=title,
        color='Color',
        color_discrete_map="identity",
        height=650 
    )
    
    # ---> THE FIX IS HERE: Force the Y-axis to ONLY use the current 10 features <---
    fig.update_yaxes(
        type='category',
        categoryorder='array',
        categoryarray=plot_df['Feature'].tolist(), # Destroys the "ghost" bars
        title=None, 
        tickfont={'size': 12}
    )
    
    fig.update_xaxes(
        title='Mathematical Impact on Final Risk Score', 
        tickfont={'size': 12}
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=40)
    )

    return pred_ui, tree_path_text, fig

if __name__ == '__main__':
    # Only generate the 10,000 sample cache if it doesn't exist yet!
    if not os.path.exists(SAMPLE_PATH):
        logger.info("First run detected: Generating Dashboard Cache...")
        preprocessor = BaselinePreprocessor()
        df_merged = load_and_engineer_features(preprocessor, dataset_type="train")
        df_merged.sample(10000, random_state=42).to_csv(SAMPLE_PATH, index=False)
        logger.info("Cache generated successfully.")
        
    logger.info("Starting Dash Server...")
    app.run(debug=True, port=8050)