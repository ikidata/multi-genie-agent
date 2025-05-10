import os
import dash
import dash_bootstrap_components as dbc
from DatabricksChatbot import DatabricksChatbot
import flask  # for request context

# Ensure environment variable is set correctly
serving_endpoint = os.getenv('SERVING_ENDPOINT')
assert serving_endpoint, 'SERVING_ENDPOINT must be set in app.yaml.'

databricks_host_secret = os.getenv('DATABRICKS_HOST_SECRET')
assert databricks_host_secret, 'DATABRICKS_HOST_SECRET must be set in app.yaml.'

databricks_token_secret = os.getenv('DATABRICKS_TOKEN_SECRET')
assert databricks_token_secret, 'DATABRICKS_TOKEN_SECRET must be set in app.yaml.'

# Initialize the Dash app with a clean theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# Instantiate chatbot without layout
chatbot = DatabricksChatbot(app=app, endpoint_name=serving_endpoint, databricks_host_secret = databricks_host_secret, databricks_token_secret = databricks_token_secret, height='600px')

# Use layout function (important!)
def serve_layout():
    try:
        user_name = flask.request.headers.get('X-Forwarded-Preferred-Username', 'Guest')
    except RuntimeError:
        user_name = 'Guest'

    return dbc.Container([
        dbc.Row([
            dbc.Col(chatbot.get_layout_for_user(user_name), width={'size': 8, 'offset': 2})
        ])
    ], fluid=True)

app.layout = serve_layout

if __name__ == '__main__':
    app.run(debug=True)