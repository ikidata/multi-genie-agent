import dash
from dash import html, Input, Output, State, dcc
import dash_bootstrap_components as dbc
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole, ExternalFunctionRequestHttpMethod
import json
import time
from openai import OpenAI
import yaml

from genie_functions import run_genie
from general_functions import call_chat_model, get_documentation
from devops_functions import create_devops_ticket

class DatabricksChatbot:
    def __init__(self, app, endpoint_name, databricks_host_secret, databricks_token_secret, height='600px'):
        self.app = app
        self.endpoint_name = endpoint_name
        self.databricks_host_secret = databricks_host_secret
        self.databricks_token_secret = databricks_token_secret
        self.height = height

        try:
            print('Initializing WorkspaceClient...')
            self.w = WorkspaceClient()
            print('WorkspaceClient initialized successfully')
        except Exception as e:
            print(f'Error initializing WorkspaceClient: {str(e)}')
            self.w = None
        
        self.get_configs()    
        self.get_authentication()
        self.layout = self._create_layout()
        self._create_callbacks()
        self._add_custom_css()


    def get_configs(self):
        '''
        Fetch necessary configs
        '''
        # Open and read the YAML file
        with open("./config.yml", 'r') as file:
            config_data = yaml.safe_load(file)

        self.system_prompt = config_data['system_prompt']
        self.devops_connection = config_data['devops_connection']
        self.genie_connection = config_data['genie_connection']
        self.tools = [value for key, value in config_data['tools'].items()]
        print("Configs were fetched successfully")


    def get_authentication(self) -> None:
        """
        Fetches authentication details from dbutils and activates the OpenAI client.

        Raises:
            RuntimeError: If any required authentication detail is missing or invalid.
        """
        try:
            self.base_url = f"https://{self.databricks_host_secret}/serving-endpoints"
            if not self.base_url:
                raise ValueError("Databricks workspace URL is missing or invalid.")

            self.openai_client = OpenAI(api_key=self.databricks_token_secret, base_url=self.base_url)

        except Exception as e:
            # Log the error and raise a runtime error with a meaningful message
            error_message = f"Failed to fetch authentication details: {str(e)}"
            raise RuntimeError(error_message)  
        print("OpenAI API client was initialized successfully")        

    def _create_layout(self):
        return html.Div([
            html.H2('Chat with multiple Genies', className='chat-title mb-3'),
            dbc.Card([
                dbc.CardBody([
                    html.Div(id='chat-history', className='chat-history'),
                ], className='d-flex flex-column chat-body')
            ], className='chat-card mb-3'),
            dbc.InputGroup([
                dbc.Input(id='user-input', placeholder='Type your message here...', type='text'),
                dbc.Button('Send', id='send-button', color='success', n_clicks=0, className='ms-2'),
                dbc.Button('Clear', id='clear-button', color='danger', n_clicks=0, className='ms-2'),
            ], className='mb-3'),
            dcc.Store(id='assistant-trigger'),
            dcc.Store(id='chat-history-store'),
            html.Div(id='dummy-output', style={'display': 'none'}),
        ], className='d-flex flex-column chat-container p-3')

    def _create_callbacks(self):
        @self.app.callback(
            Output('chat-history-store', 'data', allow_duplicate=True),
            Output('chat-history', 'children', allow_duplicate=True),
            Output('user-input', 'value'),
            Output('assistant-trigger', 'data'),
            Input('send-button', 'n_clicks'),
            Input('user-input', 'n_submit'),
            State('user-input', 'value'),
            State('chat-history-store', 'data'),
            prevent_initial_call=True
        )
        def update_chat(send_clicks, user_submit, user_input, chat_history):
            if not user_input:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update

            # Initialize chat_history with system message if it doesn't exist  
            if not chat_history:  
                chat_history = [{"role": "system", "content": self.system_prompt}]  
            else:  
                chat_history = chat_history or []  
            chat_history.append({'role': 'user', 'content': user_input})
            chat_display = self._format_chat_display(chat_history)
            chat_display.append(self._create_typing_indicator())

            return chat_history, chat_display, '', {'trigger': True}

        @self.app.callback(
            Output('chat-history-store', 'data', allow_duplicate=True),
            Output('chat-history', 'children', allow_duplicate=True),
            Input('assistant-trigger', 'data'),
            State('chat-history-store', 'data'),
            prevent_initial_call=True
        )
        def process_assistant_response(trigger, chat_history):
            if not trigger or not trigger.get('trigger'):
                return dash.no_update, dash.no_update

            chat_history = chat_history or []
            if (not chat_history or not isinstance(chat_history[-1], dict)
                    or 'role' not in chat_history[-1]
                    or chat_history[-1]['role'] != 'user'):
                return dash.no_update, dash.no_update

            try:
                assistant_response = self._call_model_endpoint(chat_history)
                chat_history.append({
                    'role': 'assistant',
                    'content': assistant_response
                })
            except Exception as e:
                error_message = f'Error: {str(e)}'
                print(error_message)  # Log the error for debugging
                chat_history.append({
                    'role': 'assistant',
                    'content': error_message
                })

            chat_display = self._format_chat_display(chat_history)
            return chat_history, chat_display

        @self.app.callback(
            Output('chat-history-store', 'data', allow_duplicate=True),
            Output('chat-history', 'children', allow_duplicate=True),
            Input('clear-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def clear_chat(n_clicks):
            print('Clearing chat')
            if n_clicks:
                return [], []
            return dash.no_update, dash.no_update

    def _call_model_endpoint(self, messages, max_tokens=750):

        function_call_messages = messages.copy()                       # Copying messages to avoid modifying the original messages
        print(f"Starting to process the next messages: {messages}")
        try:
            print('Calling model endpoint...')


            response = call_chat_model(
                        openai_client = self.openai_client,
                        model_name=self.endpoint_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        tools = self.tools
                        )

            if response.choices[0].message.tool_calls:

                # Extract message details
                message = response.choices[0].message.content           
  
                # Construct payload
                payload = {
                    "role": getattr(response.choices[0].message, "role", None),              # Safely get the role
                    "content": getattr(response.choices[0].message, "content", None),        # Safely get the content
                    "tool_calls": getattr(response.choices[0].message, "tool_calls", None),  # Safely get tool calls
                }

                # Add processes message for internal run logging
                function_call_messages.append(payload)        

                for tool_call in response.choices[0].message.tool_calls:
                    print("Tools are activated")    

                    # Parse the function arguments
                    function_arguments = json.loads(tool_call.function.arguments)

                    function_name = tool_call.function.name

                    # This part is hard coded for demo purpose only - normally would be dynamic function list
                    if "genie" in function_name:
                        print(f"Genie '{function_name }' is activated, please be patient")
                        genie_space_id = function_name.split('_')[1]

                        results = run_genie(genie_space_id = genie_space_id, 
                                    genie_conn = self.genie_connection, 
                                    prompt = function_arguments['prompt'])
                    
                    elif "devops" in function_name:
                        print("DevOps is activated, please be patient")
                        results = create_devops_ticket(content = function_arguments['content'])
                    
                    elif "documentation" in function_name:
                        print("Documentation is activated, please be patient")
                        # Fetch fabricated documentation
                        documentation_content = get_documentation()
                        messages[-1]['content'] += f"\n Context data: {documentation_content}"

                        results = call_chat_model(
                        openai_client=self.openai_client,
                        model_name=self.endpoint_name,
                        messages=messages,
                        max_tokens=500,
                        ).choices[0].message.content

                        # Remove the temporary context data from the message  
                        messages[-1]['content'] = messages[-1]['content'].replace(f"\n Context data: {documentation_content}", "")  
                    else:
                        raise Exception(f"An error occurred - function not found: {function_name}")

                    ### Adding tool call inputs as a payload to the temporary messages
                    payload = {
                        "role": "tool",
                        "content": results,  
                        "tool_call_id": tool_call.id
                    }

                    # Append the processed payload
                    function_call_messages.append(payload)

                    # Calling one more time model endpoint to clean Genie results
                    print(f"Calling model endpoint to clean the results: {function_call_messages}")
                    results = call_chat_model(
                        openai_client = self.openai_client,
                        model_name=self.endpoint_name,
                        messages=function_call_messages,
                        max_tokens=max_tokens
                        ).choices[0].message.content

            else:
                results = response.choices[0].message.content
                print('Model endpoint called successfully')
            print(f"Current messages: {messages}")
            print(f"Returning results: {results}")
            return results
        except Exception as e:
            print(f'Error calling model endpoint: {str(e)}')
            raise

    def _format_chat_display(self, chat_history):  
        return [  
            html.Div([  
                html.Div(msg['content'], className=f"chat-message {msg['role']}-message")  
            ], className=f"message-container {msg['role']}-container")  
            for msg in chat_history if isinstance(msg, dict) and 'role' in msg and msg['role'] != 'system'  
        ]  

    def _create_typing_indicator(self):
        return html.Div([
            html.Div(className='chat-message assistant-message typing-message',
                     children=[
                         html.Div(className='typing-dot'),
                         html.Div(className='typing-dot'),
                         html.Div(className='typing-dot')
                     ])
        ], className='message-container assistant-container')

    def _add_custom_css(self):
        custom_css = '''
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
        body {
            font-family: 'DM Sans', sans-serif;
            background-color: #F9F7F4; /* Oat Light */
        }
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            background-color: #FFFFFF;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .chat-title {
            font-size: 24px;
            font-weight: 700;
            color: #1B3139; /* Navy 800 */
            text-align: center;
        }
        .chat-card {
            border: none;
            background-color: #EEEDE9; /* Oat Medium */
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
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
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 16px;
            line-height: 1.4;
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
        #user-input {
            border-radius: 20px;
            border: 1px solid #DCE0E2; /* Databricks Gray - Lines */
        }
        #send-button, #clear-button {
            border-radius: 20px;
            width: 100px;
        }
        #send-button {
            background-color: #00A972; /* Databricks Green 600 */
            border-color: #00A972;
        }
        #clear-button {
            background-color: #98102A; /* Databricks Maroon 600 */
            border-color: #98102A;
        }
        .input-group {
            flex-wrap: nowrap;
        }
        '''
        self.app.index_string = self.app.index_string.replace(
            '</head>',
            f'<style>{custom_css}</style></head>'
        )

        self.app.clientside_callback(
            """
            function(children) {
                var chatHistory = document.getElementById('chat-history');
                if(chatHistory) {
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                }
                return '';
            }
            """,
            Output('dummy-output', 'children'),
            Input('chat-history', 'children'),
            prevent_initial_call=True
        )