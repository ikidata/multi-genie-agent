import os
import dash
import flask 
from flask import has_request_context, request
import dash_bootstrap_components as dbc
from DatabricksChatbot import DatabricksChatbot

# Ensure environment variable is set correctly
serving_endpoint = os.getenv('SERVING_ENDPOINT')
assert serving_endpoint, 'SERVING_ENDPOINT must be set in app.yaml.'

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Instantiate chatbot (will use layout later)
chatbot = DatabricksChatbot(
    app=app,
    endpoint_name=serving_endpoint,
    height='600px'
)

# Layout function (executed per request)
def serve_layout():
    # Only access Flask request if context is valid
    if has_request_context():
        user_name = request.headers.get('X-Forwarded-Preferred-Username', 'Guest')
    else:
        user_name = 'Guest'

    # This increases safe usage, with values set per request
    return dbc.Container([
        dbc.Row([
            dbc.Col(chatbot._create_layout(user_name), width={'size': 8, 'offset': 2})
        ])
    ], fluid=True)

# Register layout function
app.layout = serve_layout

if __name__ == '__main__':
    app.run(debug=True)