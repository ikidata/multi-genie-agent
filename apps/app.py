import os
import json
import queue
from flask import has_request_context, request, Response, stream_with_context
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


# SSE streaming status endpoint (read-only, does NOT drain the chunk queue)
@app.server.route('/api/stream/<username_key>')
def stream_events(username_key):
    """
    Server-Sent Events (SSE) endpoint for streaming status.
    This is a read-only status check - it does NOT consume chunks from the
    queue (the Dash interval callback handles that). It reports whether
    streaming is active or done based on the server-side completion flag.
    """
    def generate():
        done = chatbot.streaming_done.get(username_key, True)
        if done:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'streaming'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


if __name__ == '__main__':
    app.run(debug=True)
