import os
from flask import has_request_context, request
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import dash
from DatabricksChatbot import DatabricksChatbot
from general_functions import update_genie_spaces

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT'), 'SERVING_ENDPOINT must be set in app.yaml.'

# Initialize the Dash app with a clean theme
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.FLATLY],
                suppress_callback_exceptions=True)

# Create the chatbot component with a specified height
chatbot = DatabricksChatbot(app=app, height='600px')

# Layout function (executed per request)
def serve_layout():
    # Only access Flask request if context is valid
    if has_request_context():
        user_name = request.headers.get('X-Forwarded-Preferred-Username', 'Guest')
        obo_token = request.headers.get('X-Forwarded-Access-Token', 'Empty')
    else:
        user_name = 'Guest'
        obo_token = 'Empty'
    
    # Fetch Genie Spaces
    genie_spaces = update_genie_spaces()

    # This increases safe usage, with values set per request
    return chatbot._create_layout(user_name, obo_token, genie_spaces)

# Register layout function
app.layout = serve_layout

if __name__ == '__main__':
    app.run(debug=True)