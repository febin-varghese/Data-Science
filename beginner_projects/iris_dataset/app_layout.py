import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go


def plot_iris(data_frame, x_name, y_name):
    species = data_frame.species.unique()
    colors = ['#6BF56B', '#6B91F5', '#F97373']
    fig = go.Figure()
    x_label = " ".join(x_name.split('_'))
    y_label = " ".join(y_name.split('_'))
    layout = dict(xaxis=dict(title=x_label), yaxis=dict(title=y_label))
    for idx, s in enumerate(species):
        df_plot = data_frame[data_frame.species == s]
        fig.add_trace(go.Scatter(x=df_plot[x_name], y=df_plot[y_name], mode='markers',
                                 line=go.scatter.Line(color=colors[idx]), name=s))
    fig.update_layout(layout)
    return fig


def app_layout(df, x_label, y_label, models):
    """
    Layout of the application
    :return: Layout
    """
    variables = [{'label': c, 'value': c} for c in df.drop('species', 1).columns]
    model_options = [{'label': k, 'value': k} for k in models.keys()]

    layout = html.Div([
        html.H1("Iris Dataset", style={'textAlign': 'center', 'color': '#40D3F3'}),
        html.Div(id='plot-layer', style={'width': '80%', 'height': '100%', 'display': 'block', 'margin-left': 'auto',
                                         'margin-right': 'auto'},
                 children=[dcc.Graph(id='iris-plot', figure=plot_iris(df, x_label, y_label))]
                 ),
        html.Div(className="row", style={'width': '80%', 'height': '100%', 'display': 'block', 'margin-left': 'auto',
                                         'margin-right': 'auto'},
                 children=[
                     html.Div([dcc.Dropdown(id='x-dropdown', options=variables, value=variables[0]['value'],
                                            clearable=False)
                               ], className="six columns", style={'display': 'block'}),
                     html.Div([dcc.Dropdown(id='y-dropdown', options=variables, value=variables[1]['value'],
                                            clearable=False)
                               ], className="six columns", style={'display': 'block'}),
                     html.Div([dcc.Dropdown(id='model-selector', options=model_options, value=model_options[0]['value'],
                                            clearable=False)
                               ], className="six columns", style={'display': 'block'}),
                     html.Div([html.Button('Classify', id='classifier', n_clicks=0)
                               ], className="six columns", style={'display': 'block'}),
                 ],
                 )

    ])
    return layout
