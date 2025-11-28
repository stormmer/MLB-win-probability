"""Plotly Dash dashboard for MLB win prediction."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
from dash import dash_table
from typing import Dict, Optional, Tuple
import logging

from mlb_win_pred.config import get_config
from mlb_win_pred.utils import (
    get_data_path, get_model_path, get_report_path, setup_logger, get_team_logo_path
)
from mlb_win_pred.predict import load_model, predict_win_proba
from mlb_win_pred.theme import get_dash_theme, COLORS, get_plotly_template
from mlb_win_pred.predictions_archive import append_prediction_record, load_predictions_archive

logger = setup_logger()

# Initialize app
app = Dash(__name__)
app.title = "MLB Win Probability Dashboard"

# Load configuration
config = get_config()

# Global variables for caching
_model_data: Optional[Dict] = None
_df: Optional[pd.DataFrame] = None


def load_data_and_model() -> Tuple[pd.DataFrame, Dict]:
    """Load processed dataset and trained model.
    
    Returns:
        Tuple of (dataframe, model_data)
    """
    global _df, _model_data
    
    if _df is None:
        processed_file = get_data_path(config, "games_processed.csv", subdir="processed")
        if not processed_file.exists():
            raise FileNotFoundError(f"Processed dataset not found: {processed_file}")
        _df = pd.read_csv(processed_file)
        _df['game_date'] = pd.to_datetime(_df['game_date'])
        logger.info(f"Loaded dataset with {len(_df)} games")
    
    if _model_data is None:
        _model_data = load_model(config)
        logger.info("Loaded model")
    
    return _df, _model_data


def get_feature_importance(model_data: Dict) -> pd.DataFrame:
    """Extract feature importance from model.
    
    Args:
        model_data: Dictionary containing model and feature columns
        
    Returns:
        DataFrame with feature names and importances
    """
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return df
    else:
        # For logistic regression, return empty or use coefficients
        return pd.DataFrame({'feature': feature_cols, 'importance': [0] * len(feature_cols)})


def create_radial_gauge(probability: float) -> go.Figure:
    """Create a radial gauge chart for win probability.
    
    Args:
        probability: Win probability (0-1)
        
    Returns:
        Plotly figure
    """
    theme_template = get_plotly_template()
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Win Probability (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': COLORS["PRIMARY"]},
            'steps': [
                {'range': [0, 50], 'color': COLORS["LIGHT_GRAY"]},
                {'range': [50, 100], 'color': COLORS["MEDIUM_GRAY"]}
            ],
            'threshold': {
                'line': {'color': COLORS["SECONDARY"], 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        **theme_template
    )
    
    return fig


def create_feature_importance_chart(model_data: Dict, top_n: int = 15) -> go.Figure:
    """Create feature importance bar chart.
    
    Args:
        model_data: Dictionary containing model
        top_n: Number of top features to show
        
    Returns:
        Plotly figure
    """
    importance_df = get_feature_importance(model_data)
    top_features = importance_df.head(top_n)
    theme_template = get_plotly_template()
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker=dict(color=COLORS["PRIMARY"])
        )
    ])
    
    fig.update_layout(
        title="Top Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=200, r=20, t=40, b=20),
        **theme_template
    )
    
    return fig


def create_historical_chart(
    df: pd.DataFrame,
    team: str,
    opponent: str,
    model_data: Dict,
    limit: int = 20
) -> go.Figure:
    """Create historical performance chart with predictions.
    
    Args:
        df: Processed dataset
        team: Team abbreviation
        opponent: Opponent abbreviation
        model_data: Model data dictionary
        limit: Maximum number of games to show
        
    Returns:
        Plotly figure
    """
    # Get historical games between these teams
    historical = df[
        ((df['team'] == team) & (df['opponent'] == opponent)) |
        ((df['team'] == opponent) & (df['opponent'] == team))
    ].sort_values('game_date', ascending=False).head(limit)
    
    if historical.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No historical data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # For each game, determine if team won
    historical['team_won'] = historical.apply(
        lambda row: row['win'] if row['team'] == team else (1 - row['win']),
        axis=1
    )
    
    # Calculate predicted probabilities using model
    model = model_data['model']
    scaler = model_data.get('scaler')
    feature_cols = model_data['feature_cols']
    
    predicted_probas = []
    for idx, row in historical.iterrows():
        try:
            # Get features for this game
            X = row[feature_cols].fillna(0).values.reshape(1, -1)
            
            # Make prediction
            if scaler is not None:
                X_scaled = scaler.transform(X)
                proba = model.predict_proba(X_scaled)[0, 1]
            else:
                proba = model.predict_proba(X)[0, 1]
            
            # If this row is opponent as team, flip probability
            if row['team'] == opponent:
                proba = 1 - proba
            
            predicted_probas.append(proba)
        except Exception as e:
            logger.warning(f"Error predicting for game {idx}: {e}")
            predicted_probas.append(0.5)
    
    historical['predicted_proba'] = predicted_probas
    historical = historical.sort_values('game_date', ascending=True)
    
    fig = go.Figure()
    
    theme_template = get_plotly_template()
    
    # Add predicted probabilities as line
    fig.add_trace(go.Scatter(
        x=historical['game_date'],
        y=historical['predicted_proba'],
        mode='lines+markers',
        name='Predicted Probability',
        line=dict(color=COLORS["PRIMARY"], width=2),
        marker=dict(size=8)
    ))
    
    # Add actual outcomes as markers
    fig.add_trace(go.Scatter(
        x=historical['game_date'],
        y=historical['team_won'],
        mode='markers',
        name='Actual Outcome',
        marker=dict(
            size=12,
            color=historical['team_won'],
            colorscale='RdYlGn',
            showscale=False,
            line=dict(width=2, color=COLORS["TEXT_PRIMARY"])
        )
    ))
    
    # Add line for 50% threshold
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color=COLORS["MEDIUM_GRAY"],
        annotation_text="50%"
    )
    
    fig.update_layout(
        title=f"Historical Performance: {team} vs {opponent}",
        xaxis_title="Date",
        yaxis_title="Probability / Outcome",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(range=[0, 1]),
        **theme_template
    )
    
    return fig


# Initialize dropdowns with teams
def get_team_options():
    """Get team options for dropdowns."""
    try:
        df, _ = load_data_and_model()
        teams = sorted(df['team'].unique().tolist())
        return [{'label': team, 'value': team} for team in teams]
    except Exception as e:
        logger.warning(f"Error loading teams: {e}. Dashboard will work once data is available.")
        return []

def get_history_filter_options():
    """Get team options for history filter dropdown."""
    try:
        df, _ = load_data_and_model()
        teams = sorted(df['team'].unique().tolist())
        return [{'label': 'All Teams', 'value': 'ALL'}] + [{'label': t, 'value': t} for t in teams]
    except Exception as e:
        logger.warning(f"Error loading teams for history filter: {e}")
        return [{'label': 'All Teams', 'value': 'ALL'}]

# App layout
team_options = get_team_options()
history_filter_options = get_history_filter_options()
theme = get_dash_theme()

app.layout = html.Div([
    html.Div([
        # MLB-style header with logo
        html.Div([
            html.Div([
                # MLB Logo
                html.Img(
                    src=app.get_asset_url("mlb_logo.png"),
                    style={
                        'height': '50px',
                        'marginRight': '20px',
                        'verticalAlign': 'middle'
                    },
                    alt="MLB Logo"
                ),
                html.H1("Win Probability Predictor", style={
                    'display': 'inline-block',
                    'margin': 0,
                    'verticalAlign': 'middle',
                    'color': theme['colors']['primary'],
                    'fontSize': '28px',
                    'fontWeight': theme['layout']['font'].get('weight', '700'),
                    'fontFamily': theme['layout']['font']['family']
                })
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'padding': '20px 0',
                'borderBottom': f'1px solid {COLORS["BORDER"]}',
                'marginBottom': '30px',
                'backgroundColor': COLORS['WHITE']
            })
        ]),
        html.Div(
            id='data-status-message',
            style={
                'padding': '10px',
                'margin': '10px',
                'backgroundColor': '#fff3cd' if not team_options else 'transparent',
                'border': '1px solid #ffc107' if not team_options else 'none',
                'borderRadius': '5px',
                'textAlign': 'center'
            },
            children=[
                html.P(
                    "⚠️ Dataset not found. Please run: python -m mlb_win_pred.data_download && python -m mlb_win_pred.dataset_builder"
                ) if not team_options else html.P("✅ Data loaded successfully", style={'color': 'green'})
            ]
        ),
        
        # Sidebar
        html.Div([
            html.H3("Controls", style={'marginBottom': 20, 'color': theme['colors']['primary']}),
            
            html.Label("Team:"),
            dcc.Dropdown(
                id='team-dropdown',
                options=team_options,
                placeholder="Select team...",
                style={'marginBottom': 20}
            ),
            
            html.Label("Opponent:"),
            dcc.Dropdown(
                id='opponent-dropdown',
                options=team_options,
                placeholder="Select opponent...",
                style={'marginBottom': 20}
            ),
            
            html.Label("Game Date:"),
            dcc.DatePickerSingle(
                id='date-picker',
                placeholder="Select date...",
                style={'marginBottom': 20}
            ),
            
            html.Button(
                "Predict Win Probability",
                id='predict-button',
                n_clicks=0,
                style={
                    'width': '100%',
                    'padding': '12px',
                    'backgroundColor': COLORS['PRIMARY'],
                    'color': COLORS['WHITE'],
                    'border': 'none',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '16px',
                    'fontWeight': '600',
                    'fontFamily': theme['layout']['font']['family'],
                    'transition': 'background-color 0.2s'
                }
            ),
            
            html.Div(id='error-message', style={'color': COLORS['DANGER'], 'marginTop': 20})
            
        ], style={
            'width': '25%',
            'padding': '20px',
            'backgroundColor': COLORS['WHITE'],
            'height': '100vh',
            'position': 'fixed',
            'left': 0,
            'top': 0,
            'overflowY': 'auto',
            'borderRight': f'1px solid {COLORS["BORDER"]}',
            'boxShadow': '2px 0 4px rgba(0,0,0,0.05)'
        }),
        
        # Main content
        html.Div([
            # Team logos display
            html.Div(id='team-logos', children=html.Div("Select teams and click 'Predict Win Probability' to see logos", style={
                'textAlign': 'center',
                'color': COLORS['TEXT_MUTED'],
                'fontStyle': 'italic'
            }), style={
                'textAlign': 'center',
                'marginBottom': '20px',
                'padding': '30px',
                'backgroundColor': COLORS['WHITE'],
                'borderRadius': '8px',
                'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                'border': f'1px solid {COLORS["BORDER"]}',
                'minHeight': '120px'
            }),
            
            # Tabs for Prediction and History (MLB-style navigation)
            dcc.Tabs(id='main-tabs', value='prediction-tab', children=[
                dcc.Tab(label='Prediction', value='prediction-tab', style={
                    'fontFamily': theme['layout']['font']['family'],
                    'fontWeight': '600'
                }),
                dcc.Tab(label='Prediction History', value='history-tab', style={
                    'fontFamily': theme['layout']['font']['family'],
                    'fontWeight': '600'
                }),
            ], style={
                'marginBottom': '20px',
                'borderBottom': f'2px solid {COLORS["BORDER"]}'
            }),
            
            # Prediction tab content (always present, shown/hidden by tab selection)
            html.Div(id='prediction-tab-content', children=[
                html.Div([
                    html.Div([
                        html.H2("Win Probability", style={'marginBottom': 10, 'color': theme['colors']['primary']}),
                        html.Div(id='probability-card', style={
                            'fontSize': '48px',
                            'fontWeight': 'bold',
                            'color': theme['colors']['primary'],
                            'textAlign': 'center',
                            'padding': '20px'
                        })
                    ], style={
                        'width': '48%',
                        'display': 'inline-block',
                        'verticalAlign': 'top',
                        'backgroundColor': COLORS['WHITE'],
                        'padding': '25px',
                        'borderRadius': '8px',
                        'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                        'border': f'1px solid {COLORS["BORDER"]}',
                        'marginRight': '2%'
                    }),
                    
                    html.Div([
                        dcc.Graph(id='gauge-chart', style={'height': '300px'})
                    ], style={
                        'width': '48%',
                        'display': 'inline-block',
                        'verticalAlign': 'top',
                        'backgroundColor': COLORS['WHITE'],
                        'padding': '25px',
                        'borderRadius': '8px',
                        'border': f'1px solid {COLORS["BORDER"]}',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    })
                ], style={'marginBottom': '30px'}),
                
                html.Div([
                    html.Div([
                        dcc.Graph(id='feature-importance-chart')
                    ], style={
                        'width': '48%',
                        'display': 'inline-block',
                        'verticalAlign': 'top',
                        'backgroundColor': COLORS['WHITE'],
                        'padding': '25px',
                        'borderRadius': '8px',
                        'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                        'border': f'1px solid {COLORS["BORDER"]}',
                        'marginRight': '2%'
                    }),
                    
                    html.Div([
                        dcc.Graph(id='historical-chart')
                    ], style={
                        'width': '48%',
                        'display': 'inline-block',
                        'verticalAlign': 'top',
                        'backgroundColor': COLORS['WHITE'],
                        'padding': '25px',
                        'borderRadius': '8px',
                        'border': f'1px solid {COLORS["BORDER"]}',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                    })
                ])
            ]),
            
            # History tab content (always present, shown/hidden by tab selection)
            html.Div(id='history-tab-content', style={'display': 'none'}, children=[
                html.H3("Prediction History", style={'color': theme['colors']['primary'], 'marginBottom': '20px'}),
                html.Label("Filter by Team:"),
                dcc.Dropdown(
                    id='history-team-filter',
                    options=history_filter_options,
                    value='ALL',
                    style={'marginBottom': '20px', 'width': '300px'}
                ),
                html.Div(id='history-table-container')
            ])
            
        ], style={
            'marginLeft': '27%',
            'padding': '20px',
            'backgroundColor': COLORS['BACKGROUND_GRAY'],
            'minHeight': '100vh'
        })
    ])
], style={
    'backgroundColor': COLORS['BACKGROUND_GRAY'], 
    'minHeight': '100vh',
    'fontFamily': theme['layout']['font']['family']
})




@app.callback(
    Output('team-logos', 'children'),
    [Input('team-dropdown', 'value'),
     Input('opponent-dropdown', 'value')]
)
def update_team_logos(team, opponent):
    """Update team logos when teams are selected."""
    if not team or not opponent:
        return html.Div("Select teams to see logos", style={
            'textAlign': 'center',
            'color': COLORS['TEXT_MUTED'],
            'fontStyle': 'italic'
        })
    
    # Get team logos
    team_logo_path = get_team_logo_path(team)
    opponent_logo_path = get_team_logo_path(opponent)
    
    # Also check dashboard/assets/logos/ (Dash's default asset folder)
    dashboard_logos_dir = Path(__file__).parent / "assets" / "logos"
    team_dash_logo = dashboard_logos_dir / f"{team}.png" if dashboard_logos_dir.exists() else None
    opponent_dash_logo = dashboard_logos_dir / f"{opponent}.png" if dashboard_logos_dir.exists() else None
    
    logo_children = []
    
    # Try to use logo - check dashboard/assets first (Dash default), then project root
    team_logo_found = False
    if team_dash_logo and team_dash_logo.exists():
        try:
            logo_children.append(html.Img(
                src=app.get_asset_url(f"logos/{team}.png"),
                style={'height': '80px', 'margin': '0 20px', 'verticalAlign': 'middle'}
            ))
            team_logo_found = True
        except:
            pass
    elif team_logo_path and team_logo_path.exists():
        # Use base64 encoding for images outside Dash's asset folder
        try:
            import base64
            with open(team_logo_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
                logo_children.append(html.Img(
                    src=f"data:image/png;base64,{img_data}",
                    style={'height': '80px', 'margin': '0 20px', 'verticalAlign': 'middle'}
                ))
                team_logo_found = True
        except Exception as e:
            logger.warning(f"Could not load logo for {team}: {e}")
    
    if not team_logo_found:
        logo_children.append(html.Span(
            team, 
            style={
                'fontSize': '28px', 
                'margin': '0 20px', 
                'verticalAlign': 'middle', 
                'fontWeight': 'bold',
                'color': theme['colors']['primary']
            }
        ))
    
    logo_children.append(html.Span("vs", style={'fontSize': '20px', 'margin': '0 10px', 'verticalAlign': 'middle', 'color': COLORS['TEXT_MUTED']}))
    
    opponent_logo_found = False
    if opponent_dash_logo and opponent_dash_logo.exists():
        try:
            logo_children.append(html.Img(
                src=app.get_asset_url(f"logos/{opponent}.png"),
                style={'height': '80px', 'margin': '0 20px', 'verticalAlign': 'middle'}
            ))
            opponent_logo_found = True
        except:
            pass
    elif opponent_logo_path and opponent_logo_path.exists():
        try:
            import base64
            with open(opponent_logo_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
                logo_children.append(html.Img(
                    src=f"data:image/png;base64,{img_data}",
                    style={'height': '80px', 'margin': '0 20px', 'verticalAlign': 'middle'}
                ))
                opponent_logo_found = True
        except Exception as e:
            logger.warning(f"Could not load logo for {opponent}: {e}")
    
    if not opponent_logo_found:
        logo_children.append(html.Span(
            opponent, 
            style={
                'fontSize': '28px', 
                'margin': '0 20px', 
                'verticalAlign': 'middle', 
                'fontWeight': 'bold',
                'color': theme['colors']['primary']
            }
        ))
    
    return html.Div(logo_children, style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})


@app.callback(
    [Output('probability-card', 'children'),
     Output('gauge-chart', 'figure'),
     Output('feature-importance-chart', 'figure'),
     Output('historical-chart', 'figure'),
     Output('error-message', 'children')],
    Input('predict-button', 'n_clicks'),
    [State('team-dropdown', 'value'),
     State('opponent-dropdown', 'value'),
     State('date-picker', 'date')]
)
def update_predictions(n_clicks, team, opponent, game_date):
    """Update all charts and predictions when predict button is clicked."""
    if n_clicks == 0:
        raise PreventUpdate
    
    if not all([team, opponent, game_date]):
        return (
            "Select all fields",
            create_radial_gauge(0.5),
            create_feature_importance_chart({}),
            go.Figure(),
            "Please select team, opponent, and date"
        )
    
    try:
        # Load data and model
        df, model_data = load_data_and_model()
        
        # Predict probability
        proba = predict_win_proba(team, opponent, game_date, config)
        
        # Append to archive
        try:
            append_prediction_record(
                team=team,
                opponent=opponent,
                game_date=game_date,
                predicted_prob=proba,
                actual_result=None,
                source="dash",
                config=config
            )
        except Exception as e:
            logger.warning(f"Failed to append to archive: {e}")
        
        # Format probability
        proba_text = f"{proba:.1%}"
        
        # Create charts
        gauge_fig = create_radial_gauge(proba)
        feature_fig = create_feature_importance_chart(model_data)
        historical_fig = create_historical_chart(df, team, opponent, model_data)
        
        return proba_text, gauge_fig, feature_fig, historical_fig, ""
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        return (
            "Error",
            create_radial_gauge(0.5),
            create_feature_importance_chart({}),
            go.Figure(),
            error_msg
        )


@app.callback(
    [Output('prediction-tab-content', 'style'),
     Output('history-tab-content', 'style')],
    Input('main-tabs', 'value')
)
def update_tab_visibility(active_tab):
    """Show/hide tab content based on selected tab."""
    if active_tab == 'prediction-tab':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}


@app.callback(
    [Output('history-team-filter', 'options'),
     Output('history-table-container', 'children')],
    [Input('main-tabs', 'value'),
     Input('history-team-filter', 'value')]
)
def update_history_content(active_tab, history_filter):
    """Update history table content and filter options."""
    # Only update if history tab is active
    if active_tab != 'history-tab':
        raise PreventUpdate
    
    # Get team options for filter
    try:
        df, _ = load_data_and_model()
        teams = sorted(df['team'].unique().tolist())
        team_filter_options = [{'label': 'All Teams', 'value': 'ALL'}] + [{'label': t, 'value': t} for t in teams]
    except:
        team_filter_options = [{'label': 'All Teams', 'value': 'ALL'}]
    
    # Load archive
    archive_df = load_predictions_archive(config)
    
    if archive_df.empty:
        return team_filter_options, html.P("No predictions recorded yet.")
    
    # Filter by team if specified
    if history_filter and history_filter != 'ALL':
        archive_df = archive_df[
            (archive_df['team'] == history_filter) | 
            (archive_df['opponent'] == history_filter)
        ]
    
    # Get most recent 50
    archive_df = archive_df.tail(50).sort_values('timestamp', ascending=False)
    
    # Format for display
    display_df = archive_df[['timestamp', 'team', 'opponent', 'game_date', 'predicted_prob', 'source']].copy()
    display_df['predicted_prob'] = display_df['predicted_prob'].apply(lambda x: f"{x:.1%}")
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['game_date'] = display_df['game_date'].dt.strftime('%Y-%m-%d')
    
    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[
            {'name': 'Timestamp', 'id': 'timestamp'},
            {'name': 'Team', 'id': 'team'},
            {'name': 'Opponent', 'id': 'opponent'},
            {'name': 'Game Date', 'id': 'game_date'},
            {'name': 'Predicted Prob', 'id': 'predicted_prob'},
            {'name': 'Source', 'id': 'source'},
        ],
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': theme['layout']['font']['family']
        },
        style_header={
            'backgroundColor': theme['colors']['primary'],
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': COLORS['LIGHT_GRAY']
            }
        ],
        page_size=20,
        sort_action='native'
    )
    
    return team_filter_options, table


if __name__ == "__main__":
    app.run(debug=True, port=8050)

