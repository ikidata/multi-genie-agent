from dash import html, Input, Output, State, dcc, callback_context
import dash_bootstrap_components as dbc
import dash
from pydantic import BaseModel, ValidationError, Field, field_validator
from databricks.sdk import WorkspaceClient
import json
import time
from openai import OpenAI
import yaml
import os 
from flask import session
import logging
import uuid
import threading
import re
import queue  
import mlflow

from logger import activate_logger
from general_functions import call_chat_model, extract_assistant_and_tool_messages, get_model_list, clean_text, optimize_conversation_history, count_tokens 
from genie_functions import get_genie_name_by_id
from agent import ChatBotAgent 
from simple_agent import SimpleChatBotAgent

#########################################################
#
####  Deactivating MLFlow tracing
#
# MLFlow is deactivated since this agent is not deployed
# via Mosaic AI model serving. The rason for that is that
# Databrcicks Apps doesn't support OBO for agents (yet?)
#########################################################
mlflow.openai.autolog(disable=True)


#### Validators
class ChatConfig(BaseModel):
    """Configuration model for chat application."""
    system_prompt: str = Field(..., min_length=10)
    react_system_prompt: str = Field(..., min_length=10)
                               
    @field_validator('system_prompt')
    @classmethod
    def validate_system_prompt(cls, v):
        if not v.strip():
            raise ValueError("System prompt cannot be empty or just whitespace")
        return v
    
    @field_validator('react_system_prompt')
    @classmethod
    def validate_react_system_prompt(cls, v):
        if not v.strip():
            raise ValueError("React system prompt cannot be empty or just whitespace")
        return v

class DatabricksChatbot:
    def __init__(self, app, height='600px'):
        self.app = app
        self.height = height

        # Set up logging
        self.logger = activate_logger()

        try:
            self.logger.info(f'Initializing WorkspaceClient...')
            self.w = WorkspaceClient()
            self.logger.info(f'WorkspaceClient initialized successfully')
        except Exception as e:
            self.logger.info(f'Error initializing WorkspaceClient: {str(e)}')
            self.w = None
        
        self.get_configs()    
        self.get_authentication()
        self._create_callbacks()
        self._add_custom_css()

        # Instance-level dictionaries for user data
        self.chunk_queues = {}     # Per-user queue for streaming response chunks  
        self.user_threads = {}     # username_key -> threading.Thread  
        self.user_queues = {}      # username_key -> queue.Queue  
        self.user_locks = {}       # username_key -> threading.Lock (to protect thread start)    

    def get_configs(self):
        """
        Set up configs - basically just system prompt currently.
        """
        try:
            with open("./config.yml", 'r') as file:
                config_data = yaml.safe_load(file)

            # Validate config using Pydantic
            validated_config = ChatConfig(**config_data)
            
            # Set class attributes from validated config
            self.system_prompt = validated_config.system_prompt
            self.react_system_prompt = validated_config.react_system_prompt

        except FileNotFoundError:
            raise FileNotFoundError("Config file not found: ./config.yml")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except ValueError as e:
            raise ValueError(f"Configuration validation error: {e}")

    def get_authentication(self) -> None:
        """
        Fetches authentication using SDK and activates the OpenAI client.

        Raises:
            RuntimeError: If any required authentication detail is missing or invalid.
        """
        try:
            self.openai_client = self.w.serving_endpoints.get_open_ai_client()
        except Exception as e:
            # Log the error and raise a runtime error with a meaningful message
            raise RuntimeError(f"Failed to fetch authentication details: {str(e)}")  
        self.logger.info("OpenAI API client was initialized successfully")        

    def _create_layout(self, user_name: str, obo_token: str, genie_spaces: dict):
        '''
        This function defines the complete layout of the app
        '''

        # Use username as the session identifier instead of generating a UUID
        username_key = user_name.lower()
    
        # Reset user thread and queue data when starting a new session
        if username_key in self.user_threads:
            # Clean up any running thread for this user
            if self.user_threads[username_key].is_alive():
                # Can't actually stop the thread, but it can be removed from dictionary
                self.logger.info(f"Clearing previous thread for user {username_key}")
            del self.user_threads[username_key]
        
        # Clear the queue for this user
        if username_key in self.chunk_queues:
            self.logger.info(f"Clearing previous chunk queue for user {username_key}")
            self.chunk_queues[username_key] = queue.Queue()  

        user_name_cleaned = user_name.split('@')[0].capitalize()   # Cleaning user name
        default_message = f"Hello {user_name_cleaned}! Welcome to the Agentic Multi-Genie solution on Databricks 🤖 On the right, you’ll find options for fine-tuning your agent — including selecting Active Genie Spaces, which LLM model to use, Activating ReAct and choosing max output tokens. You can also view the current message history on the right.\n\n💡Start utilizing your internal team of Genies for smarter, faster and automated data analysis."

        # Fetch LLM model list
        llm_model_list = get_model_list()
        
        return dbc.Container([
            # Top row with header and user badge
            dbc.Row([
                # Left: Chat title spanning the chatbox width
                dbc.Col(
                    html.H2(f'Agentic Multi-Genie Assistant 🧞‍♂️', className='chat-title'),
                    width=9,
                    className="chat-title-col"
                ),
                # Right: User badge
                dbc.Col(
                    html.Div([
                        html.Span("👤 ", className="user-icon"),
                        html.Span(f"User: {user_name_cleaned}", className="user-name")
                    ], className="user-info-badge"),
                    width=3,
                    className="user-badge-col"
                )
            ], className="mb-3 g-0 header-row"),
            
            # Main content row
            dbc.Row([
                # Chatbot column (left)
                dbc.Col([
                    # Chat card
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.Div(default_message, className='chat-message assistant-message', style={'whiteSpace': 'pre-line'})
                            ], id='chat-history', className='chat-history'),
                        ], className='d-flex flex-column chat-body')
                    ], className='chat-card mb-3'),
                    
                    # Input area
                    dbc.InputGroup([
                        dbc.Input(id='user-input', placeholder='Type your message here...', type='text'),
                        dbc.Button('Send', id='send-button', color='success', n_clicks=0, className='ms-2'),
                        dbc.Button('Clear', id='clear-button', color='danger', n_clicks=0, className='ms-2'),
                    ], className='mb-3'),
                ], xs=12, sm=12, md=9, className="chat-column"),
                
                # Settings column (right)
                dbc.Col([
                    # Options panel
                    html.Div([
                        # Centered header
                        html.H4("Options", className="mb-3 text-center fw-bold"),

                        # Section title - made larger and bold
                        html.Div("Activate Genie Spaces", className="h6 fw-bold mb-2"),

                        # Icon + Dropdown row for Genie Spaces
                        html.Div([
                            html.Div([
                                html.Img(src="/assets/genie_small.png", style={'height': '40px', 'width': 'auto'})
                            ], className="me-3 d-flex align-items-center"),

                            # Dropdown on the right - with improved width
                            html.Div([
                                dcc.Dropdown(
                                    id='genie-spaces-dropdown',
                                    options=[{'label': space, 'value': space} for space in genie_spaces],
                                    placeholder="Select Genie Spaces",
                                    multi=True,
                                    style={'width': '100%'}
                                )
                            ], className="flex-grow-1 d-flex align-items-center", style={'width': '100%'})
                        ], className="d-flex align-items-center w-100 mb-3"),

                        # Divider
                        html.Hr(className="my-3"),
                        
                        # Model dropdown with larger, bold label and icon
                        html.Div("Model", className="h6 fw-bold mb-2"),
                        html.Div([
                            html.Div([
                                html.Img(src="/assets/llm_model_small.png", style={'height': '40px', 'width': 'auto'})
                            ], className="me-3 d-flex align-items-center"),
                            
                            html.Div([
                                dcc.Dropdown(
                                    id='model-dropdown',
                                    options=llm_model_list,
                                    value='databricks-claude-3-7-sonnet',
                                    style={'width': '100%'}
                                )
                            ], className="flex-grow-1 d-flex align-items-center", style={'width': '100%'})
                        ], className="d-flex align-items-center w-100 mb-3"),
                        
                        # Divider
                        html.Hr(className="my-3"),

                        # Max tokens input with larger, bold label and icon
                        html.Div("Max Tokens", className="h6 fw-bold mb-2"),
                        html.Div([
                            html.Div([
                                html.Img(src="/assets/llm_max_tokens_small.png", style={'height': '40px', 'width': 'auto'})
                            ], className="me-3 d-flex align-items-center"),
                            
                            html.Div([
                                dbc.Input(
                                    id='max-tokens-input',
                                    type='number',
                                    value=1000,
                                    min=100,
                                    max=4000,
                                    step=100,
                                    size="sm",
                                    style={'width': '100%'}
                                )
                            ], className="flex-grow-1 d-flex align-items-center", style={'width': '100%'})
                        ], className="d-flex align-items-center w-100 mb-3"),

                        # Divider
                        html.Hr(className="my-3"),

                        # ReAct Agent 
                        html.Div("Activate ReAct agent", className="h6 fw-bold mb-2"),  
                        html.Div([  
                            html.Div([  
                                html.Img(src="/assets/llm_model_react_small.png", style={'height': '40px', 'width': 'auto'})  
                            ], className="me-3"),  
                            html.Div([  
                                dcc.RadioItems(  
                                    id='react-radio',  
                                    options=[  
                                        {'label': 'Yes', 'value': 'yes'},  
                                        {'label': 'No', 'value': 'no'}  
                                    ],  
                                    value='no',  
                                    labelStyle={'display': 'inline-block', 'margin-right': '10px'},  
                                    inputStyle={'margin-right': '5px'}  
                                )  
                            ], className="flex-grow-1", style={'width': '100%'})  
                        ], className="d-flex align-items-center w-100 mb-3"),  
                        
                        # Apply button
                        dbc.Button("Apply", id="apply-settings", color="primary", size="sm", className="w-100 mt-2")
                    ], className='settings-panel p-3 bg-white rounded shadow-sm'),
                    
                    # Stats panel below the options panel
                    html.Div([
                        # Centered header
                        html.H4("Chat Stats", className="mb-2 text-center fw-bold"),
                        
                        # Messages and tokens on the same row (left and right)
                        html.Div([
                            # Left side - Messages count
                            html.Div([
                                html.Div("Messages: ", className="stats-label"),
                                html.Div("0", id="message-count", className="stats-value")
                            ], className="d-flex align-items-center"),
                            
                            # Spacer
                            html.Div(className="flex-grow-1"),
                            
                            # Right side - Token count
                            html.Div([
                                html.Div("Tokens: ~", className="stats-label"),
                                html.Div("0", id="token-count", className="stats-value")
                            ], className="d-flex align-items-center")
                        ], className="d-flex justify-content-between w-100")
                        
                    ], className='stats-panel p-2 bg-white rounded shadow-sm mt-3')
                ], xs=12, sm=12, md=3, className="options-column")
            ], className="g-0 content-row"),
            
            # Add interval for streaming updates
            dcc.Interval(id='streaming-interval', interval=200, disabled=True),  ## Adding a bit extra latency due to possible loading slowness
            
            # Store for session-specific data
            dcc.Store(id='session-id', data=username_key),
            dcc.Store(id='chat-history-store'),
            dcc.Store(id='streaming-state', data={
                'streaming': False,
                'chunks': [],
                'session_id': username_key
            }),
            dcc.Store(id='processing-flag', data=False),
            dcc.Store(id='user-name-store', data=user_name),
            
            # Store for Genie Spaces data
            dcc.Store(id='genie-spaces-store', data=genie_spaces),
            
            # Store for OBO token
            dcc.Store(id='obo-token-store', data=obo_token),

            # Store for user settings with default parameters
            dcc.Store(id='llm-params-store', data={
                'model': 'databricks-claude-3-7-sonnet',
                'react_agent': 'no',
                'max_tokens': 1000,
                'genie_spaces': []
            }),
            
            # Interval to update Genie Spaces
            dcc.Interval(id='genie-spaces-interval', interval=3000, max_intervals=1),
            
            html.Div(id='dummy-output', style={'display': 'none'}),

            # Add the small Ikidata logo at the bottom left
            html.Div([
                html.Img(src="/assets/Ikidata_aurora_small.png", style={'height': '20px', 'width': 'auto'})
            ], style={
                'position': 'fixed',
                'bottom': '10px',
                'left': '10px',
                'zIndex': '1000'
            })
            
        ], fluid=True, className="p-2 d-flex flex-column chat-container")


    def _start_user_worker(self, username_key):  
        """Start the per-user streaming worker if not already running."""  
        if username_key not in self.user_queues:  
            self.user_queues[username_key] = queue.Queue()  
        if username_key not in self.user_locks:  
            self.user_locks[username_key] = threading.Lock()  
    
        lock = self.user_locks[username_key]  
    
        with lock:  
            thread = self.user_threads.get(username_key)  
            if thread and thread.is_alive():  
                # Already running  
                return  
    
            def worker():  
                while True:  
                    request = self.user_queues[username_key].get()  
                    if request is None:  # Sentinel to stop thread  
                        break  
                    chat_history, user_name, obo_token, llm_params = request  
                    try:  
                        self._streaming_worker(chat_history, username_key, user_name, obo_token, llm_params)  
                    except Exception as e:  
                        self.logger.error(f"Exception in streaming_worker {username_key}: {e}")  
                    self.user_queues[username_key].task_done()  
    
                self.logger.info(f"Worker thread for {username_key} exiting.")  
    
            thread = threading.Thread(target=worker, daemon=True)  
            thread.start()  
            self.user_threads[username_key] = thread  

    def _streaming_worker(self, messages: list, username_key: str, user_name: str, obo_token:str, llm_params: dict = None):
        """Background worker to fetch chunks and add them to the queue"""
        try:  
            # Use default LLM parameters if missing  
            if not llm_params:  
                llm_params = {  
                    'model': 'databricks-claude-3-7-sonnet',  
                    'react_agent': 'no',  
                    'max_tokens': 1000,  
                    'genie_spaces': {}  
                }  
    
            # Ensure chunk queue exists — created when user started streaming  
            chunk_queue = self.chunk_queues.get(username_key)  
            if chunk_queue is None:  
                # Defensive: create a queue (rare), but ideally managed elsewhere  
                chunk_queue = queue.Queue()  
                self.chunk_queues[username_key] = chunk_queue  
    
            chunk_count = 0  
            # Stream chunks from your model endpoint (blocking call)  
            for chunk in self._call_model_endpoint_streaming(  
                messages=messages,  
                user_name=user_name,  
                obo_token=obo_token,  
                model=llm_params.get('model'),  
                react_agent=llm_params.get('react_agent'),  
                max_tokens=llm_params.get('max_tokens'),  
                genie_spaces=llm_params.get('genie_spaces')  
            ):  
                if chunk:  
                    self.chunk_queues[username_key].put(chunk)  
                    chunk_count += 1  
    
            self.logger.info(f"[{user_name}] Streaming complete, added {chunk_count} chunks")  
            # Signal streaming completion with a special sentinel  
            self.chunk_queues[username_key].put("__STREAMING_COMPLETE__")  
    
        except Exception as e:  
            self.logger.error(f"[{user_name}] Error in streaming worker: {str(e)}")  
            self.chunk_queues[username_key].put(f"ERROR: {str(e)}")  
            self.chunk_queues[username_key].put("__STREAMING_COMPLETE__")  

    def _create_callbacks(self):
        @self.app.callback(
            Output('chat-history', 'children'),
            Output('chat-history-store', 'data'),
            Output('user-input', 'value'),
            Output('streaming-state', 'data'),
            Output('streaming-interval', 'disabled'),
            Output('processing-flag', 'data'),
            Input('send-button', 'n_clicks'),
            Input('user-input', 'n_submit'),
            State('user-input', 'value'),
            State('chat-history-store', 'data'),
            State('session-id', 'data'),
            State('user-name-store', 'data'),
            State('obo-token-store', 'data'),
            State('llm-params-store', 'data'),  # Added user settings state
            prevent_initial_call=True
        )
        
        def update_chat(send_clicks, user_submit, user_input, chat_history, username_key, user_name, obo_token, llm_params):  
            """
            Updates the chat conversation by appending the user input, starts a background thread to stream the assistant's response and manages the chat display and streaming state for real-time updates.
            """
            if not user_input:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

            # Initialize chat_history with system message if it doesn't exist  
            if not chat_history:  
                chat_history = [{"role": "system", "content": self.system_prompt}]  
            else:  
                chat_history = chat_history or []  
            
            # Add user message
            chat_history.append({'role': 'user', 'content': user_input})
            
            # Format the chat display with user message
            chat_display = self._format_chat_display(chat_history)
            chat_display.append(self._create_typing_indicator())  
        
            # Ensure per-user worker thread is running  
            self._start_user_worker(username_key)  
        
            # Enqueue the request for streaming  
            self.user_queues[username_key].put((chat_history, user_name, obo_token, llm_params))  
        
            streaming_state = {  
                'streaming': True,  
                'chunks': [],  
                'session_id': username_key  
            }  

            # Don't clear input field yet, disabling input meanwhile  
            return chat_display, chat_history, '', streaming_state, False, True  

        @self.app.callback(
            Output('chat-history', 'children', allow_duplicate=True),
            Output('chat-history-store', 'data', allow_duplicate=True),
            Output('streaming-state', 'data', allow_duplicate=True),
            Output('streaming-interval', 'disabled', allow_duplicate=True),
            Output('processing-flag', 'data', allow_duplicate=True),
            Input('streaming-interval', 'n_intervals'),
            State('streaming-state', 'data'),
            State('chat-history', 'children'),
            State('chat-history-store', 'data'),
            State('session-id', 'data'), 
            State('user-name-store', 'data'),
            prevent_initial_call=True
        )

        def update_streaming(n_intervals, streaming_state, chat_display, chat_history, username_key, user_name):
            """
            Handles periodic updates of the streaming assistant response by processing received chunks, updating the chat display and history, managing typing indicators, handling completion and timeout conditions, and ensuring proper synchronization and UI feedback during the streaming process.
            """
            # Create a deep copy of streaming_state to avoid reference issues
            streaming_state = dict(streaming_state) if streaming_state else {}

            # If streaming is already marked as complete or completion was already handled,
            # immediately disable the interval and return without any further processing
            if not streaming_state.get('streaming') or streaming_state.get('completion_handled'):
                return dash.no_update, dash.no_update, streaming_state, True, False
            
            # Get the thread-safe queue for current user or create a new one if it doesn't exist
            chunk_queue = self.chunk_queues.get(username_key)
            if not chunk_queue:
                self.logger.debug(f"[{user_name}] Creating new queue for user {username_key}")
                self.chunk_queues[username_key] = queue.Queue()
                chunk_queue = self.chunk_queues[username_key]

            chunks_to_process = []  
        
            if chunk_queue:  
                try:  
                    while True:  
                        chunk = chunk_queue.get_nowait()  
                        chunks_to_process.append(chunk)  
                except queue.Empty:  
                    pass 
            
            # Check for streaming complete marker in the queue
            streaming_complete = "__STREAMING_COMPLETE__" in chunks_to_process
            
            # Remove completion marker for processing  
            chunks_to_display = [c for c in chunks_to_process if c != "__STREAMING_COMPLETE__"]  
        
            # Copy the current chat display or initialize  
            new_chat_display = chat_display.copy() if chat_display else []  

            # Check if typing indicator exists and remove it
            typing_indicator_removed = False
            typing_indicator_index = None
            for i, div in enumerate(new_chat_display):
                if 'typing-message' in str(div):
                    new_chat_display.pop(i)
                    typing_indicator_removed = True
                    typing_indicator_index = i
                    break
            

            # Process streaming complete marker first before any forced completion
            if streaming_complete:
                # Mark that completion has been handled to avoid race conditions
                streaming_state['completion_handled'] = True
                streaming_state['streaming'] = False
                
                # Collect chunks to display - combine them for better markdown rendering
                combined_content = ""
                genie_chunks = []
                
                for chunk in chunks_to_display:
                    # Special handling for Genie chunks
                    if chunk.startswith("🧞‍♂️ Genie") and chunk.endswith("activated"):
                        genie_chunks.append(chunk)
                    else:
                        # Combine regular content chunks
                        combined_content += chunk

                # If combined content exists, add it to chat history and display
                if combined_content:
                    if chat_history is None:
                        chat_history = []
                    
                    # Add the combined content as a single message
                    chat_history.append({'role': 'assistant', 'content': combined_content})
                    
                    # Generate a unique key for this content
                    content_key = f"{time.time()}"
                    
                    # Create a message container with Markdown support
                    content_container = html.Div([
                        dcc.Markdown(
                            combined_content,
                            className="chat-message assistant-message chunk-message",
                            dangerously_allow_html=True
                        )
                    ], className="message-container assistant-container", id=f"chunk-{content_key}")
                    
                    # Add the content to the display
                    new_chat_display.append(content_container)
                
                # Now handle any Genie chunks separately
                for genie_chunk in genie_chunks:
                    # Check if this Genie is already in the display to avoid duplicates
                    genie_exists = False
                    for div in new_chat_display:
                        if genie_chunk in str(div):
                            genie_exists = True
                            break
                    
                    if not genie_exists:
                        # Create a special message container with GIF for tool call
                        tool_call_container = html.Div([
                            html.Div([
                                html.Img(src="/assets/genie_loading.gif", style={'height': '100px', 'width': 'auto'}),
                                html.Div(genie_chunk, style={'marginTop': '10px'})
                            ], className="chat-message assistant-message chunk-message", 
                            style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
                        ], className="message-container assistant-container", id=f"tool-call-{time.time()}")
                        
                        # Add the special container to the display
                        new_chat_display.append(tool_call_container)
                        
                        # Also add to chat history
                        if chat_history is None:
                            chat_history = []
                        chat_history.append({'role': 'assistant', 'content': genie_chunk})
                
                # Find any tool call GIF boxes and convert them to success messages
                i = 0
                while i < len(new_chat_display):
                    if '/assets/genie_loading.gif' in str(new_chat_display[i]):
                        # Using regex to find Genie ID
                        try:
                            genie_name = (re.search(r"🧞\u200d♂️ Genie '([^']+)' activated", json.dumps(new_chat_display[i], ensure_ascii=False)) or [None, ""])[1]
                        except:
                            genie_name = "missing"

                        # Replace with success message
                        normal_container = html.Div([
                            html.Div(f"🧞 Genie '{genie_name}' processed successfully", className="chat-message assistant-message chunk-message")
                        ], className="message-container assistant-container")
                        
                        new_chat_display[i] = normal_container
                    i += 1
                
                self.logger.info(f"[{user_name}] Streaming completed, disabling interval")
                
                # Optimize conversation history
                if chat_history:
                    original_stats = {
                        'length': len(chat_history),
                        'tokens': count_tokens(chat_history)
                    }
                    
                    # Optimize the conversation history
                    chat_history = optimize_conversation_history(chat_history)
                    
                    # Get updated stats
                    updated_stats = {
                        'length': len(chat_history),
                        'tokens': count_tokens(chat_history)
                    }
                    
                    # Log the results optimization process results
                    if original_stats['tokens'] != updated_stats['tokens']:
                        self.logger.info(
                            f"[{user_name}] Chat history optimized: {original_stats['length']} → {updated_stats['length']} messages, "
                            f"{original_stats['tokens']} → {updated_stats['tokens']} tokens"
                        )
                    else:
                        self.logger.info(
                            f"[{user_name}] Current history: {updated_stats['length']} messages, {updated_stats['tokens']} tokens"
                        )

            else:
                # Get the current time and the last update time
                current_time = time.time()
                last_update_time = streaming_state.get('last_update_time', current_time) 
                
                # Update the last update time if we have new chunks
                if chunks_to_process:
                    streaming_state['last_update_time'] = current_time
                
                # Check if there are any active Genie operations (loading GIFs) in the chat display
                has_active_genie = False
                for div in new_chat_display:
                    if '/assets/genie_loading.gif' in str(div):
                        has_active_genie = True
                        break
                
                # Adding safety timeout - only consider timeout if:
                # 1. There are no active Genie operations
                # 2. It's been more than 60 seconds since the last update (even with active Genies)
                short_timeout = 5  # 5 seconds for normal operations
                long_timeout = 60  # 60 seconds even with active Genies

                timed_out = False
                if has_active_genie:
                    # Use long timeout for active Genie operations
                    timed_out = (current_time - last_update_time > long_timeout)
                    if timed_out:
                        self.logger.info(f"Forcing stream completion for {username_key} - Genie operation timed out after {long_timeout} seconds")
                else:
                    # Use short timeout for normal operations
                    timed_out = (current_time - last_update_time > short_timeout)
                    if timed_out:
                        self.logger.info(f"Forcing stream completion for {username_key} - timed out after {short_timeout} seconds")
                
                # If timed out, force completion
                if timed_out:
                    streaming_state['streaming'] = False
                    streaming_state['completion_handled'] = True
                    
                    # Process any remaining chunks
                    chunks_to_display = [chunk for chunk in chunks_to_process if chunk != "__STREAMING_COMPLETE__"]
                    
                    # Collect chunks to display - combine them for better Markdown rendering
                    combined_content = ""
                    genie_chunks = []
                    
                    for chunk in chunks_to_display:
                        # Special handling for Genie chunks
                        if chunk.startswith("🧞‍♂️ Genie") and chunk.endswith("activated"):
                            genie_chunks.append(chunk)
                        else:
                            # Combine regular content chunks
                            combined_content += chunk
                    
                    # If we have combined content, add it to chat history and display
                    if combined_content:
                        if chat_history is None:
                            chat_history = []
                        
                        # Add the combined content as a single message
                        chat_history.append({'role': 'assistant', 'content': combined_content})
                        
                        # Generate a unique key for this content
                        content_key = f"{time.time()}"
                        
                        # Create a message container with Markdown support
                        content_container = html.Div([
                            dcc.Markdown(
                                combined_content,
                                className="chat-message assistant-message chunk-message",
                                dangerously_allow_html=True
                            )
                        ], className="message-container assistant-container", id=f"chunk-{content_key}")
                        
                        # Add the content to the display
                        new_chat_display.append(content_container)
                    
                    # Now handle any Genie chunks separately
                    for genie_chunk in genie_chunks:
                        # Check if this Genie is already in the display to avoid duplicates
                        genie_exists = False
                        for div in new_chat_display:
                            if genie_chunk in str(div):
                                genie_exists = True
                                break
                        
                        if not genie_exists:
                            # Create a special message container with GIF for tool call
                            tool_call_container = html.Div([
                                html.Div([
                                    html.Img(src="/assets/genie_loading.gif", style={'height': '100px', 'width': 'auto'}),
                                    html.Div(genie_chunk, style={'marginTop': '10px'})
                                ], className="chat-message assistant-message chunk-message", 
                                style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
                            ], className="message-container assistant-container", id=f"tool-call-{time.time()}")
                            
                            # Add the special container to the display
                            new_chat_display.append(tool_call_container)
                            
                            # Also add to chat history
                            if chat_history is None:
                                chat_history = []
                            chat_history.append({'role': 'assistant', 'content': genie_chunk})
                    
                    # Add fallback message if typing indicator was removed and no content was added
                    if typing_indicator_removed and not combined_content and not genie_chunks:
                        fallback_message = html.Div([
                            dcc.Markdown(
                                "Sorry, I got lost in my thoughts. Could you please repeat your question?",
                                className="chat-message assistant-message"
                            )
                        ], className="message-container assistant-container")
                        
                        new_chat_display.append(fallback_message)
                        
                        # Also add to chat history
                        if chat_history is None:
                            chat_history = []
                        chat_history.append({
                            'role': 'assistant', 
                            'content': "Sorry, I got lost in my thoughts. Could you please repeat your question?"
                        })
                        
                        self.logger.info(f"Added fallback message for {username_key}")
                    
                    # Convert any remaining genie loading GIFs based on whether we had to add a fallback
                    i = 0
                    while i < len(new_chat_display):
                        if '/assets/genie_loading.gif' in str(new_chat_display[i]):
                            try:
                                genie_name = (re.search(r"🧞\u200d♂️ Genie '([^']+)' activated", json.dumps(new_chat_display[i], ensure_ascii=False)) or [None, ""])[1]
                            except:
                                genie_name = "missing"

                            # If we had to add a fallback message, show interrupted; otherwise show success
                            if typing_indicator_removed and not combined_content and not genie_chunks:
                                message = f"🧞 Genie '{genie_name}' operation was interrupted"
                            else:
                                message = f"🧞 Genie '{genie_name}' processed successfully"

                            normal_container = html.Div([
                                html.Div(message, className="chat-message assistant-message chunk-message")
                            ], className="message-container assistant-container")
                            
                            new_chat_display[i] = normal_container
                            self.logger.info(f"Converted Genie loading GIF to {'interrupted' if 'interrupted' in message else 'success'} message for {username_key}")
                        i += 1
                    
                    # Log conversation stats
                    if chat_history:
                        self.logger.info(f"[{user_name}] Current history: {len(chat_history)} messages, {count_tokens(chat_history)} tokens")
                else:
                    # Normal streaming update - process chunks and add typing indicator
                    chunks_to_display = [chunk for chunk in chunks_to_process if chunk != "__STREAMING_COMPLETE__"]

                    # Collect chunks to display - combine them for better Markdown rendering
                    combined_content = ""
                    genie_chunks = []
                    
                    for chunk in chunks_to_display:
                        # Special handling for Genie chunks
                        if chunk.startswith("🧞‍♂️ Genie") and chunk.endswith("activated"):
                            genie_chunks.append(chunk)
                        else:
                            # Combine regular content chunks
                            combined_content += chunk
                    
                    # If we have combined content, add it to chat history and display
                    if combined_content:
                        if chat_history is None:
                            chat_history = []
                        
                        # Add the combined content as a single message
                        chat_history.append({'role': 'assistant', 'content': combined_content})
                        
                        # Generate a unique key for this content
                        content_key = f"{time.time()}"
                        
                        # Create a message container with Markdown support
                        content_container = html.Div([
                            dcc.Markdown(
                                combined_content,
                                className="chat-message assistant-message chunk-message",
                                dangerously_allow_html=True
                            )
                        ], className="message-container assistant-container", id=f"chunk-{content_key}")
                        
                        # Add the content to the display
                        new_chat_display.append(content_container)
                    
                    # Now handle any Genie chunks separately
                    for genie_chunk in genie_chunks:
                        # Check if this Genie is already in the display to avoid duplicates
                        genie_exists = False
                        for div in new_chat_display:
                            if genie_chunk in str(div):
                                genie_exists = True
                                break
                        
                        if not genie_exists:
                            # Create a special message container with GIF for tool call
                            tool_call_container = html.Div([
                                html.Div([
                                    html.Img(src="/assets/genie_loading.gif", style={'height': '100px', 'width': 'auto'}),
                                    html.Div(genie_chunk, style={'marginTop': '10px'})
                                ], className="chat-message assistant-message chunk-message", 
                                style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
                            ], className="message-container assistant-container", id=f"tool-call-{time.time()}")
                            
                            # Add the special container to the display
                            new_chat_display.append(tool_call_container)
                            
                            # Also add to chat history
                            if chat_history is None:
                                chat_history = []
                            chat_history.append({'role': 'assistant', 'content': genie_chunk})
                    
                    # Add typing indicator back if we're still streaming
                    new_chat_display.append(self._create_typing_indicator())
            
            return new_chat_display, chat_history, streaming_state, False, True

        @self.app.callback(
            Output('user-input', 'disabled'),
            Output('send-button', 'disabled'),
            Input('processing-flag', 'data')
        )
        def toggle_input_disable(is_processing):
            """
            Toggles disabling of the input field and send button based on whether the agent is currently processing a response.
            """
            return is_processing, is_processing

        @self.app.callback(
            Output('chat-history-store', 'data', allow_duplicate=True),
            Output('chat-history', 'children', allow_duplicate=True),
            Output('streaming-state', 'data', allow_duplicate=True),
            Output('streaming-interval', 'disabled', allow_duplicate=True),
            Input('clear-button', 'n_clicks'),
            State('session-id', 'data'),
            State('user-name-store', 'data'),
            prevent_initial_call=True
        )
        def clear_chat(n_clicks, username_key, user_name):
            """
            Clears the chat history and message queues for a user, resets the streaming state and logs the action when the clear chat button is clicked
            """
            if n_clicks:
                # Clean up any running thread for this user
                if username_key in self.user_threads and self.user_threads[username_key].is_alive():
                    # Can't actually stop the thread, but can remove it from the dictionary
                    del self.user_threads[username_key]
                
                # Clear the queue for this user
                if username_key in self.chunk_queues:
                    self.chunk_queues[username_key] = queue.Queue()  
                
                # Reset streaming state
                streaming_state = {
                    'streaming': False,
                    'chunks': [],
                    'session_id': username_key
                }
                self.logger.info(f"[{user_name}] Cleared chat and message history for user {username_key}")
                return [], [], streaming_state, True
            self.logger.info(f"[{user_name}] Cleared chat and message history for user {username_key}")
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Added callback for getting parameter values from options
        @self.app.callback(
            [Output('llm-params-store', 'data'),
            Output('apply-settings', 'children', allow_duplicate=True)],
            Input('apply-settings', 'n_clicks'),
            State('model-dropdown', 'value'),
            State('react-radio', 'value'),
            State('max-tokens-input', 'value'),
            State('genie-spaces-dropdown', 'value'),
            State('session-id', 'data'),
            State('user-name-store', 'data'),
            State('genie-spaces-store', 'data'),  
            prevent_initial_call=True
        )
        def apply_settings(n_clicks, model, react_agent, max_tokens, genie_spaces, username_key, user_name, genie_spaces_all):
            """
            Applies and returns the user-selected model settings and filters genie spaces, logging the updated configuration upon the apply button click
            """
            if not n_clicks:
                return dash.no_update, dash.no_update
            
            # Filter genie spaces 
            if genie_spaces:
                active_genie_spaces = {space: genie_spaces_all[space] for space in genie_spaces if space in genie_spaces_all}
            else:
                active_genie_spaces = {}

            # Create a dictionary with all the settings
            settings = {
                'model': model or 'databricks-claude-3-7-sonnet',          
                'react_agent': react_agent,                        
                'max_tokens': max_tokens,                       
                'genie_spaces': active_genie_spaces                     
            }

            # Log the settings for debugging
            self.logger.info(f"[{user_name}] Applied settings for user {username_key}: {settings}")
            
            # Return settings and update button text to indicate success
            return settings, "✓ Applied!"

        # Add another callback to reset the button text after a delay
        @self.app.callback(
            Output('apply-settings', 'children', allow_duplicate=True),
            Input('apply-settings', 'children'),
            prevent_initial_call=True
        )
        def reset_button_text(text):
            """
            Resets the apply settings button text to its default label shortly after showing a success confirmation.
            """
            if text == "✓ Applied!":
                time.sleep(1)  # Wait 1 seconds
                return "Apply Settings"
            return dash.no_update
        
        @self.app.callback(
        [Output('message-count', 'children'),
         Output('token-count', 'children')],
        [Input('chat-history-store', 'data')]
    )
        def update_chat_stats(chat_history):
            """
            Updates and returns the message count and token count for the chat history, excluding system message from message count.
            """
            if not chat_history:
                return "0", "0"
            
            # Count messages (excluding system messages)
            message_count = sum(1 for msg in chat_history if msg.get('role') != 'system')
            
            # Estimate token count
            token_count = count_tokens(chat_history)
            
            return str(message_count), str(token_count)

    def _call_model_endpoint_streaming(self, messages: list, user_name: str, obo_token: str, model: str, react_agent: str, max_tokens: int, genie_spaces: dict[str, dict[str, str]]) -> list:
        """
        Calls the model endpoint to stream responses, handling and logging various event types including content, tool calls, and errors, while yielding cleaned or informative output strings.
        """
        try:
            self.logger.info(f"[{user_name}] Calling model endpoint with {len(messages)} messages using model {model} and ReAct agent is '{react_agent}'")

            # Set up the agent with parameters
            if react_agent == 'yes':                
                agent = ChatBotAgent(model_name=model, max_tokens=max_tokens, obo_token=obo_token, genie_spaces=genie_spaces)
                
                # Updating ReAct system prompt
                react_messages = [{**msg, 'content': self.react_system_prompt} if msg.get('role') == 'system' else msg for msg in messages] 

                # Stream the response
                for event in agent.predict_stream({"messages": react_messages}):                
                    if event.delta.role == 'tool':
                        self.logger.info(f"[{user_name}] |Agent| Tool event: {event.delta}")
                        yield ""  # or some status message  
                        continue
                    elif event.delta.content and event.delta.content != '-' and event.delta.tool_calls:
                        self.logger.info(f"[{user_name}] |Agent| Content with tool calls: {event.delta.content}")
                        yield clean_text(event.delta.content)  # Cleans ReAct extra instruction texts
                        for tc in event.delta.tool_calls:
                            if "genie_" in tc.function.name:
                                genie_name = get_genie_name_by_id(genie_spaces, tc.function.name.split("genie_")[1])
                                self.logger.info(f"[{user_name}] |Agent| 🧞‍♂️ Genie '{genie_name}' activated")
                                yield f"🧞‍♂️ Genie '{genie_name}' activated"
                    elif event.delta.tool_calls:
                        self.logger.info(f"[{user_name}] |Agent| Tool calls only: {event.delta.tool_calls}")
                        for tc in event.delta.tool_calls:
                            if "genie_" in tc.function.name:
                                genie_name = get_genie_name_by_id(genie_spaces, tc.function.name.split("genie_")[1])
                                self.logger.info(f"[{user_name}] |Agent| 🧞‍♂️ Genie '{genie_name}' activated")
                                yield f"🧞‍♂️ Genie '{genie_name}' activated"
                    elif event.delta.content and event.delta.content != '-':
                        self.logger.info(f"[{user_name}] |Agent| Content only: {event.delta.content}")
                        yield clean_text(event.delta.content)  # Cleans ReAct extra instruction texts
                    else:
                        self.logger.warning(f"[{user_name}] |Agent| Unexpected event format: {event}")
                        yield "Sorry, couldn't process the request"

            else:
                agent = SimpleChatBotAgent(model_name=model, max_tokens=max_tokens, obo_token=obo_token, genie_spaces=genie_spaces) 

                # Stream the response
                for event in agent.predict_stream({"messages": messages}):                
                    if event.delta.role == 'tool':
                        self.logger.info(f"[{user_name}] |Agent| Tool event: {event.delta}")
                        yield ""  # or some status message  
                        continue
                    elif event.delta.tool_calls:
                        self.logger.info(f"[{user_name}] |Agent| Tool calls only: {event.delta.tool_calls}")
                        for tc in event.delta.tool_calls:
                            if "genie_" in tc.function.name:
                                genie_name = get_genie_name_by_id(genie_spaces, tc.function.name.split("genie_")[1])
                                yield f"🧞‍♂️ Genie '{genie_name}' activated"
                    elif event.delta.content and event.delta.content != '-':
                        self.logger.info(f"[{user_name}] |Agent| Content only: {event.delta.content}")
                        yield event.delta.content
                    else:
                        self.logger.warning(f"[{user_name}] |Agent| Unexpected event format: {event}")
                        yield "Sorry, couldn't process the request"          


        except Exception as e:
            self.logger.error(f"[{user_name}] Error calling model endpoint: {str(e)}")
            yield f"ERROR: {str(e)}"

    def _format_chat_display(self, chat_history):
        formatted_messages = []
        
        for msg in chat_history:
            if isinstance(msg, dict) and 'role' in msg and msg['role'] != 'system':
                if msg['role'] == 'assistant':
                    # Use Dash's Markdown component for assistant messages with enhanced options
                    message_content = dcc.Markdown(
                        msg['content'],
                        className=f"chat-message {msg['role']}-message",
                        dangerously_allow_html=True,  # Allow HTML in Markdown
                        # Add any other Markdown options that might help with complex formatting
                        style={'whiteSpace': 'pre-wrap'}  # Preserve whitespace
                    )
                else:
                    # For user messages, keep as plain text
                    message_content = html.Div(
                        msg['content'],
                        className=f"chat-message {msg['role']}-message"
                    )
                
                formatted_messages.append(
                    html.Div(
                        message_content,
                        className=f"message-container {msg['role']}-container"
                    )
                )
        
        return formatted_messages


    def _create_typing_indicator(self): 
        """
        Creates a typing indicator component with animated dots to show that the assistant is typing.
        """ 
        return html.Div([  
            html.Div(className='chat-message assistant-message typing-message', children=[  
                html.Span(className='typing-dot bounce'),  
                html.Span(className='typing-dot bounce', style={'animationDelay': '0.2s'}),  
                html.Span(className='typing-dot bounce', style={'animationDelay': '0.4s'}),  
            ])  
        ], className='message-container assistant-container')
    
    def _add_custom_css(self):
        """
        Adds custom CSS styles to the app to control formatting and enhance the appearance and functionality of the entire application. And some Javascript improvements.
        """
        custom_css = '''
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
        
        body {
            font-family: 'DM Sans', sans-serif;
            background-color: #EDEBE8; /* Background color */
        }
                
        /* Chat container styles */
        .chat-container {
            max-width: 70% !important;
            margin: 0 auto;
            background-color: #EDEBE8; /* Background color */
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15); /* Enhanced shadow */
            height: 100vh;
            max-height: 100vh; /* Ensure it doesn't exceed viewport height */
            display: flex;
            flex-direction: column;
            border: 1.5px solid rgba(0, 0, 0, 0.7); /* Lighter border */
            padding: 20px;
            overflow-y: hidden; /* Prevent scrolling of container */
        }
        
        /* Row alignment fix */
        .row {
            align-items: flex-start; /* Align items to the top */
        }
        
        /* User info badge - aligned with chat header */
        .user-info-badge {
            background: linear-gradient(135deg, #1B3139, #2D4550);
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 5px;
            width: 100%;
            height: 100%;
            margin-right: -7px; /* Move 5px closer to the right edge */
            transition: all 0.3s ease;
            z-index: 10; /* Ensure it's above other elements */
        }
        
        .user-icon {
            font-size: 16px;
        }
        
        .user-name {
            white-space: nowrap;
        }
        
        /* Header row styling */
        .header-row {
            display: flex;
            align-items: center;
        }

        /* Chat title column */
        .chat-title-col {
            padding-right: 0;
        }

        /* User badge column */
        .user-badge-col {
            padding-left: 0;
            padding-right: 7px; /* Reduce right padding by 7px */
            display: flex;
            align-items: center;
            justify-content: flex-end;
        }

        /* Chat title styling */
        .chat-title {
            font-size: 28px;
            font-weight: 700;
            color: white;
            text-align: center;
            margin: 0 auto;
            padding: 15px 30px;
            border-radius: 15px;
            background: linear-gradient(135deg, #1B3139, #2D4550);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            position: relative;
            overflow: hidden;
            width: 100%;
        }
        
        /* Add decorative elements to the header */
        .chat-title:before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
            opacity: 0.6;
            z-index: 0;
        }

        /* Updated chat card with gradient border effect like input area */
        .chat-card {
            border: none !important; /* Remove default border */
            background-color: #FFFFFF; /* White background */
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            border-radius: 20px; /* Rounded corners */
            min-height: 400px;
            height: calc(100vh - 200px); /* Fixed calculation instead of vh */
            max-height: calc(100vh - 200px); /* Maximum height */
            width: 100%;
            margin: 0;
            position: relative;
            background: linear-gradient(#FFFFFF, #FFFFFF) padding-box,
                    linear-gradient(145deg, rgba(0,0,0,0.8), rgba(0,0,0,0.2)) border-box;
            border: 1.5px solid transparent !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
        }
                            
        .chat-body {
            flex-grow: 1;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        
        .chat-history {
            flex-grow: 1;
            overflow-y: auto;
            padding: 15px;
        }
        
        .message-container {
            display: flex;
            margin-bottom: 15px;
        }
        
        .user-container {
            justify-content: flex-end;
        }
        
        .chat-message {
            max-width: 80%;
            padding: 12px 18px; /* Slightly larger padding */
            border-radius: 20px;
            font-size: 16px;
            line-height: 1.4;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        }
        
        .user-message {
            background-color: #FF3621; /* Databricks Orange 600 */
            color: white;
        }
        
        .assistant-message {
            background-color: #1B3139; /* Databricks Navy 800 */
            color: white;
        }
        
        .typing-message {
            background-color: #2D4550; /* Lighter shade of Navy 800 */
            color: #EEEDE9; /* Oat Medium */
            display: flex;
            justify-content: center;
            align-items: center;
            min-width: 60px;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            background-color: #EEEDE9; /* Oat Medium */
            border-radius: 50%;
            margin: 0 3px;
            animation: typing-animation 1.4s infinite ease-in-out;
        }
        
        .typing-dot:nth-child(1) { animation-delay: 0s; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing-animation {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-5px); }
            100% { transform: translateY(0px); }
        }
        
        /* Enhanced Input Area */
        .input-group {
            flex-wrap: nowrap;
            margin: 15px 0 20px 0 !important;
            padding: 0 5px;
            position: relative;
            background: linear-gradient(145deg, #ffffff, #f0f0f0);
            border-radius: 25px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            padding: 5px;
            transition: all 0.3s ease;
        }
        
        .input-group:focus-within {
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }
        
        /* Updated input with light border */
        #user-input {
            border-radius: 25px;
            border: 1px solid rgba(0, 0, 0, 0.1); /* Light border added */
            padding: 15px 25px;
            font-size: 16px;
            background-color: transparent;
            box-shadow: none;
            transition: all 0.3s ease;
            flex-grow: 1;
        }
        
        #user-input:focus {
            outline: none;
            border-color: rgba(0, 0, 0, 0.2); /* Slightly darker border on focus */
        }
        
        #user-input::placeholder {
            color: #888;
            font-style: italic;
            transition: all 0.3s ease;
        }
        
        #user-input:focus::placeholder {
            opacity: 0.5;
            transform: translateX(10px);
        }
        
        /* Enhanced Buttons */
        #send-button, #clear-button {
            border-radius: 25px;
            padding: 12px 25px;
            font-weight: 600;
            font-size: 15px;
            border: none;
            margin-left: 8px;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            z-index: 1;
        }
        
        #send-button:before, #clear-button:before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 0%;
            height: 100%;
            background: rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
            z-index: -1;
        }
        
        #send-button:hover:before, #clear-button:hover:before {
            width: 100%;
        }
        
        #send-button {
            background: linear-gradient(135deg, #00A972, #008f60);
            box-shadow: 0 4px 10px rgba(0, 169, 114, 0.3);
        }
        
        #send-button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 6px 15px rgba(0, 169, 114, 0.4);
        }
        
        #send-button:active {
            transform: translateY(1px);
        }
        
        #clear-button {
            background: linear-gradient(135deg, #98102A, #800d24);
            box-shadow: 0 4px 10px rgba(152, 16, 42, 0.3);
        }
        
        #clear-button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 6px 15px rgba(152, 16, 42, 0.4);
        }
        
        #clear-button:active {
            transform: translateY(1px);
        }
        
        /* Newly added improved markdown CSS additions to enable custom CSS in chat response */
        .assistant-message h1, .assistant-message h2, .assistant-message h3 {
            margin-top: 10px;
            margin-bottom: 10px;
            color: white;
        }

        .assistant-message h1 {
            font-size: 22px;
        }

        .assistant-message h2 {
            font-size: 20px;
        }

        .assistant-message h3 {
            font-size: 18px;
        }

        .assistant-message strong {
            font-weight: bold;
        }

        .assistant-message br {
            margin-bottom: 5px;
        }

        /* Style for code blocks if needed */
        .assistant-message code {
            background-color: #3d4f57;
            padding: 2px 4px;
            border-radius: 3px;
        }

        .assistant-message pre {
            background-color: #3d4f57;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 10px 0;
        }

        /* Options column positioning - no top margin needed now */
        .options-column {
            display: flex;
            flex-direction: column;
            height: 100%;
            padding: 0 !important;
            padding-left: 20px !important;
            margin-top: 0px; /* No margin needed since user box is in a separate row */
        }

        /* Settings panel styles */
        .settings-panel {
            height: auto;
            font-size: 0.9rem;
            padding: 20px;
            background-color: #FFFFFF; /* White background */
            border-radius: 20px; /* Consistent with chat area */
            border: none;
            max-height: calc(100vh - 250px); /* Adjusted to leave room for stats panel */
            overflow-y: auto;
            width: 100%;
            min-height: 500px; /* Increased from 450px */
            position: relative;
            background: linear-gradient(#FFFFFF, #FFFFFF) padding-box,
                    linear-gradient(145deg, rgba(0,0,0,0.8), rgba(0,0,0,0.2)) border-box;
            border: 1.5px solid transparent !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
            margin-top: 0px; 
        }
                                                    
        /* Make option headers stand out */
        .h6.fw-bold {
            font-size: 1rem;
            font-weight: 600 !important;
            color: #333;
        }
        
        /* Make the dropdown text clearer */
        .Select-control, .Select-menu-outer {
            font-size: 0.9rem;
            border-radius: 15px;
        }
        
        /* Ensure dropdowns take full width */
        .Select {
            width: 100% !important;
        }
        
        /* Fix dropdown menu positioning */
        .Select-menu-outer {
            width: 100% !important;
            min-width: 100% !important;
            border-radius: 15px;
        }
        
        /* Center the header text */
        .text-center {
            text-align: center !important;
        }
        
        /* Style for horizontal rule */
        hr.my-3 {
            margin-top: 1rem;
            margin-bottom: 1rem;
            opacity: 0.25;
        }
        
        /* Style for multi-select dropdown */
        .Select--multi .Select-value {
            background-color: #f0f0f0;
            border-radius: 15px;
            padding: 1px 8px;
            margin: 2px;
            font-size: 0.8rem;
        }
        
        /* Style for the remove icon */
        #genie-spaces-dropdown .Select-value-icon,
        #rag-dropdown .Select-value-icon {
            border: none;
            padding: 0 5px;
            border-right: none;
        }
        
        /* Fix for dropdown arrow positioning */
        #genie-spaces-dropdown .Select-arrow-zone,
        #rag-dropdown .Select-arrow-zone {
            position: absolute;
            right: 0;
            top: 0;
            height: 100%;
            display: flex;
            align-items: center;
            padding-right: 8px;
        }
        
        /* Fix for input field in multi-select */
        #genie-spaces-dropdown .Select-input,
        #rag-dropdown .Select-input {
            height: 28px;
            padding-left: 5px;
        }
        
        /* Keep dropdown container fixed height */
        #genie-spaces-dropdown .Select-control,
        #rag-dropdown .Select-control {
            height: 36px !important;
            overflow: hidden;
        }
        
        /* Make the multi-value wrapper scrollable instead of expanding */
        #genie-spaces-dropdown .Select-multi-value-wrapper,
        #rag-dropdown .Select-multi-value-wrapper {
            max-height: 36px;
            overflow-y: auto;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
        }
        
        /* Make selected values smaller */
        #genie-spaces-dropdown .Select-value,
        #rag-dropdown .Select-value {
            font-size: 0.7rem;
            padding: 0 5px;
            height: 22px;
            line-height: 22px;
            margin: 2px;
        }
        
        /* Make the value label text smaller */
        #genie-spaces-dropdown .Select-value-label,
        #rag-dropdown .Select-value-label {
            max-width: 60px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        /* Style for icons */
        .genie-icon {
            height: 20px;
            width: auto;
        }
        
        /* Flex alignment for icon and label */
        .d-flex {
            display: flex;
        }
        
        .align-items-center {
            align-items: center;
        }
        
        .me-3 {
            margin-right: 1rem;
        }
        
        .w-100 {
            width: 100% !important;
        }
        
        .align-middle {
            vertical-align: middle;
        }
        
        /* Slider specific styles */
        .rc-slider {
            width: 100% !important;
            margin: 0;
        }
        
        /* Fix the slider track */
        .rc-slider-track {
            background-color: #007bff;
        }
        
        /* Fix the slider handle */
        .rc-slider-handle {
            border-radius: 50%;
        }
        
        /* Fix the slider marks */
        .rc-slider-mark-text {
            font-size: 0.8rem;
        }
        
        /* Input elements more round */
        .form-control, .form-control-sm {
            border-radius: 15px;
        }
        
        /* Button elements more round */
        .btn {
            border-radius: 15px;
        }
        
        /* Stats panel styles */
        .stats-panel {
            height: auto;
            font-size: 0.85rem;
            padding: 10px !important;
            background-color: #FFFFFF;
            border-radius: 20px;
            border: none;
            width: 100%;
            position: relative;
            background: linear-gradient(#FFFFFF, #FFFFFF) padding-box,
                    linear-gradient(145deg, rgba(0,0,0,0.8), rgba(0,0,0,0.2)) border-box;
            border: 1.5px solid transparent !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
            margin-top: 10px;
            margin-bottom: 10px;
        }

        .stats-panel h4 {
            font-size: 1rem;
            margin-bottom: 0.5rem !important;
        }

        .stats-label {
            font-weight: 600;
            margin-right: 5px;
            font-size: 0.85rem;
        }

        .stats-value {
            font-family: monospace;
            background-color: #f0f0f0;
            padding: 1px 6px;
            border-radius: 10px;
            font-weight: 500;
            font-size: 0.85rem;
        }

        /* Apply button styling - WOW effect */
        #apply-settings {
            background: linear-gradient(135deg, #007bff, #0056b3);
            border: none;
            font-weight: 600;
            padding: 10px 20px;
            border-radius: 20px;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 4px 10px rgba(0, 123, 255, 0.3);
            position: relative;
            overflow: hidden;
            z-index: 1;
        }
        
        #apply-settings:before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 0%;
            height: 100%;
            background: rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
            z-index: -1;
        }
        
        #apply-settings:hover:before {
            width: 100%;
        }
        
        #apply-settings:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 6px 15px rgba(0, 123, 255, 0.4);
        }
        
        #apply-settings:active {
            transform: translateY(1px);
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .chat-container {
                max-width: 100% !important;
                margin-right: auto;
                padding: 10px; /* Smaller padding on mobile */
            }
            
            .chat-title {
                font-size: 22px;
                padding: 12px 20px;
            }
            
            .settings-panel {
                margin-top: 10px;
                border-left: none;
                padding-top: 15px;
                padding-left: 15px;
                height: auto;
            }
            
            .chat-card {
                height: 50vh; /* Smaller on mobile */
                min-height: 300px;
            }
            
            .user-info-badge {
                left: 35px;
                bottom: 5px;
                padding: 3px 10px;
                font-size: 12px;
            }
            
            .options-column {
                margin-top: 0; /* Reset margin on mobile */
            }
        }
        '''
            
        self.app.index_string = self.app.index_string.replace(
            '</head>',
            f'<style>{custom_css}</style></head>'
        )

        # Fixed JavaScript with proper syntax and simplified functionality
        enhanced_js = """
        <script>
        // Global variable to track if user is manually scrolling
        let userIsScrolling = false;
        let messageCount = 0;
        
        // Function to add tooltips to dropdown values
        function enhanceMultiSelect() {
            document.querySelectorAll('#genie-spaces-dropdown .Select-value-label, #rag-dropdown .Select-value-label').forEach(label => {
                if (label && label.textContent) {
                    label.setAttribute('title', label.textContent);
                }
            });
        }
        
        // Function to scroll chat to bottom
        function scrollChatToBottom(force = false) {
            const chatHistory = document.getElementById('chat-history');
            if (chatHistory) {
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }
        }
        
        // Function to check if new messages have been added
        function checkForNewMessages() {
            const chatHistory = document.getElementById('chat-history');
            if (chatHistory) {
                const currentMessageCount = chatHistory.children.length;
                if (currentMessageCount > messageCount) {
                    // New message detected, scroll to bottom
                    scrollChatToBottom();
                    messageCount = currentMessageCount;
                }
            }
        }
        
        // Monitor for send button clicks
        function setupSendButtonListener() {
            const sendButton = document.getElementById('send-button');
            if (sendButton) {
                sendButton.addEventListener('click', function() {
                    // Short delay to ensure message is added to the DOM
                    setTimeout(scrollChatToBottom, 100);
                });
            }
            
            // Also monitor Enter key in the input field
            const userInput = document.getElementById('user-input');
            if (userInput) {
                userInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        // Short delay to ensure message is added to the DOM
                        setTimeout(scrollChatToBottom, 100);
                    }
                });
            }
        }
        
        // Run on load
        window.addEventListener('load', function() {
            enhanceMultiSelect();
            
            // Initial scroll to bottom
            scrollChatToBottom();
            
            // Setup event listeners
            setupSendButtonListener();
            
            // Initialize message count
            const chatHistory = document.getElementById('chat-history');
            if (chatHistory) {
                messageCount = chatHistory.children.length;
            }
            
            // Set up scroll event listeners
            if (chatHistory) {
                chatHistory.addEventListener('scroll', function() {
                    // Allow free scrolling
                    userIsScrolling = true;
                });
            }
        });
        
        // Periodically check for new messages (every 500ms)
        setInterval(checkForNewMessages, 500);
        
        // Run dropdown enhancement less frequently
        setInterval(enhanceMultiSelect, 2000);
        </script>
        """
        
        self.app.index_string = self.app.index_string.replace(
            '</body>',
            f'{enhanced_js}</body>'
        )

        # Update the clientside callback to always scroll to bottom when messages change
        self.app.clientside_callback(
            """
            function(children) {
                // Use a short timeout to ensure DOM is updated
                setTimeout(function() {
                    var chatHistory = document.getElementById('chat-history');
                    if(chatHistory) {
                        chatHistory.scrollTop = chatHistory.scrollHeight;
                    }
                }, 50);
                return '';
            }
            """,
            Output('dummy-output', 'children'),
            Input('chat-history', 'children'),
            prevent_initial_call=True
        )