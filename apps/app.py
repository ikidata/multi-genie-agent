import os
import dash
import dash_bootstrap_components as dbc
from DatabricksChatbot import DatabricksChatbot

# Ensure environment variable is set correctly
serving_endpoint = os.getenv('SERVING_ENDPOINT')
assert serving_endpoint, 'SERVING_ENDPOINT must be set in app.yaml.'

databricks_host_secret = os.getenv('DATABRICKS_HOST_SECRET')
assert databricks_host_secret, 'DATABRICKS_HOST_SECRET must be set in app.yaml.'

databricks_token_secret = os.getenv('DATABRICKS_TOKEN_SECRET')
assert databricks_token_secret, 'DATABRICKS_TOKEN_SECRET must be set in app.yaml.'

# Initialize the Dash app with a clean theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Create the chatbot component with a specified height
chatbot = DatabricksChatbot(app=app, endpoint_name=serving_endpoint, databricks_host_secret = databricks_host_secret, databricks_token_secret = databricks_token_secret, height='600px')

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(chatbot.layout, width={'size': 8, 'offset': 2})
    ])
], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True)