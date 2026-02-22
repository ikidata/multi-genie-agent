import re
import requests
import json 
import time
import yaml
from pydantic import BaseModel, Field, field_validator
from mlflow.entities import SpanType
from typing import List, Generator, Any, Optional, Dict, Union
import tiktoken ##0.7.0
import mlflow
from mlflow.types.agent import ChatAgentMessage

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from flask import request

#### Validators
class ToolException(Exception):
    """Custom exception for tool-related errors."""
    pass

class ToolFunctionCall(BaseModel):
    name: str
    arguments: str 

    def get_parsed_arguments(self) -> Dict[str, Any]:
        return json.loads(self.arguments)
    
class ToolCall(BaseModel):
    id: str
    type: str
    function: ToolFunctionCall

class ToolCallsOutput(BaseModel):
    tool_calls: List[ToolCall]

class ChatConfig(BaseModel):
    """Configuration model for chat application."""
    model_name: str = Field(..., min_length=2)

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v):
        if not v.strip():
            raise ValueError("Model name cannot be empty or just whitespace")
        return v



def create_tool_calls_output(results: object) -> dict:
    tool_calls = []

    for tool_call in results.tool_calls:
        function_arguments = tool_call.function.arguments
        if isinstance(function_arguments, dict):
            function_arguments = json.dumps(function_arguments)

        tool_call_model = ToolCall(
            id=tool_call.id,
            type=tool_call.type,
            function=ToolFunctionCall(
                name=tool_call.function.name,
                arguments=function_arguments 
            )
        )
        tool_calls.append(tool_call_model)

    # Convert to native dict so it can be injected into ChatAgentMessage
    return ToolCallsOutput(tool_calls=tool_calls).model_dump()["tool_calls"]

def prepare_messages_for_llm(messages: list[ChatAgentMessage]) -> list[dict[str, Any]]:
    """Filter out ChatAgentMessage fields that are not compatible with LLM message formats"""
    compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
    prepared = []
    for m in messages:
        if hasattr(m, 'model_dump_compat'):
            raw = m.model_dump_compat(exclude_none=True)
        elif hasattr(m, 'model_dump'):
            raw = {k: v for k, v in m.model_dump().items() if v is not None}
        elif isinstance(m, dict):
            raw = {k: v for k, v in m.items() if v is not None}
        else:
            raw = {k: v for k, v in m.__dict__.items() if v is not None}
        prepared.append({k: v for k, v in raw.items() if k in compatible_keys})
    return prepared

#@mlflow.trace(name="Call LLM model", span_type='LLM')
def call_chat_model(openai_client: any, model_name: str, messages: list, temperature: float = 0.1, max_tokens: int = 1000, **kwargs):
    """
    Calls the chat model and returns the response text or tool calls.

    Parameters:
        message (list): Message to send to the chat model.
        temperature (float, optional): Controls response randomness. Defaults to 0.1.
        max_tokens (int, optional): Maximum tokens for the response. Defaults to 750.
        **kwargs: Additional parameters for the chat model.

    Returns:
        str: Response from the chat model.
    """
    # Check if 'tool' parameter exists and is an empty list, remove it if so
    if 'tool' in kwargs and isinstance(kwargs['tool'], list) and not kwargs['tool']:
        del kwargs['tool']

    # Prepare arguments for the chat model
    chat_args = {
        "model": model_name,  
        "messages": prepare_messages_for_llm(messages),
        "max_tokens": max_tokens,
    #    "temperature": temperature,      <--- Deactivated since not the all newest models support temperature anymore
        **kwargs,  # Update with additional arguments
    }
    try:
        chat_completion = openai_client.chat.completions.create(**chat_args)
        return chat_completion  
    except Exception as e:
        error_message = f"Model endpoint calling error: {e}"
        print(error_message)
        raise RuntimeError(error_message)

def run_rest_api(server_hostname: str, token: str, api_version: str, api_command: str, action_type: str,  payload: dict = {}) -> str:
    """
    Run Databricks REST API endpoint dynamically.
    
    This function makes a request to a specified Databricks REST API endpoint using the provided parameters.
    
    Args:
        server_hostname (str): The Databricks server hostname, e.g., "https://adb-123456789.1.azuredatabricks.com"
        token (str): The Databricks personal access token for authentication
        api_version (str): The API version, e.g., "2.0"
        api_command (str): The specific API command/endpoint, e.g., "jobs/list"
        action_type (str): The HTTP method to use, either 'POST' or 'GET'
        payload (dict, optional): The request payload data. Defaults to an empty dict.
    
    Returns:
        requests.Response: The response object from the API call if successful
        Exception: The exception object if the API call fails
    
    Raises:
        AssertionError: If action_type is not 'POST' or 'GET'
    """
    try:
        assert action_type in ['POST', 'GET'], f'Only POST and GET are supported but you used {action_type}'
        url = f"{server_hostname}/api/{api_version}/{api_command}"
        headers = {'Authorization': 'Bearer %s' % token}
        session = requests.Session()
        
        resp = session.request(action_type, url, data=json.dumps(payload), verify=True, headers=headers)

        return resp
    except Exception as e:
        return e

def clean_text(text: str) -> str:
    """
    Cleans the input text by extracting and returning the content after the '**ANSWER**' marker,
    where the marker may or may not be followed by whitespace and/or newline,
    and removing any other marker keywords ('**THOUGHT**', '**ACTION**', '**OBSERVATION**', '**ANSWER**') from the text.

    Steps:
    1. Searches for the '**ANSWER**' marker optionally followed by whitespace/newline(s).
       - If found, keeps only the content after '**ANSWER**' and any following whitespace/newlines.
    2. Removes any remaining markers '**THOUGHT**', '**ACTION**', '**OBSERVATION**', or '**ANSWER**' from the text.

    Args:
        text (str): The raw text containing various markers and content.

    Returns:
        str: The cleaned text with only the answer content, free of marker keywords.
    """
    try:
        # Input validation
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
        
        if not text:
            return ""
        
        # Extract everything after '**ANSWER**' and optional whitespace/newlines
        answer_match = re.search(r'\*\*ANSWER\*\*(?::|)\s*(?:\n|)(.*)', text, re.DOTALL)
        if answer_match:
            text = answer_match.group(1)
        
        # Remove any remaining markers without removing their content
        cleaned_text = re.sub(r'\*\*(THOUGHT|ACTION|OBSERVATION|ANSWER)\*\*(?::|)\s*(?:\n|)', '', text)
        
        # Additional cleaning - handle nested or malformed markers. Sometimes markers might be incomplete or nested
        additional_cleanup_pattern = r'\*\*(?:THOUGHT|ACTION|OBSERVATION|ANSWER)(?:\*\*)?'
        cleaned_text = re.sub(additional_cleanup_pattern, '', cleaned_text)
        
        # Remove any leading/trailing whitespace
        final_text = cleaned_text.strip()
    except:
        final_text = text
    return final_text


def create_tools_from_genie_spaces(genie_spaces: dict[str, dict[str, str]]) -> list[dict]:
    """
    Creates a list of tool definitions for OpenAI function calling from Genie spaces.
    
    This function converts a dictionary of Genie spaces into a list of tool definitions
    that can be used with OpenAI's function calling API. Each Genie space is converted
    into a tool with a standardized structure.
    
    Args:
        genie_spaces (dict[str, dict[str, str]]): Dictionary of Genie spaces where:
            - The key is the name of the Genie space
            - The value is a dictionary containing at least:
                - 'id': A unique identifier for the Genie space
                - 'description': A description of the Genie space (can be empty)
    
    Returns:
        list[dict]: A list of tool definitions compatible with OpenAI's function calling API.
        Each tool has:
            - type: "function"
            - function: Contains name, description, and parameters
    """
    tools = []  # Initialize empty list to store tool definitions
    
    for name, details in genie_spaces.items():
        # Extract the Genie space ID from details
        genie_id = details['id']
        
        # Use the provided description or generate a fallback description if empty
        description = details['description'] if details['description'] else f"Genie space for {name}"
        
        # Create a standardized tool name using the Genie ID
        tool_name = f"genie_{genie_id}"
        
        # Create the tool definition and append to the list
        tools.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Write an optimized prompt to fetch the requested information from Genie Space."
                        }
                    },
                    "required": ["prompt"],
                }
            }
        })
    
    return tools

def extract_assistant_and_tool_messages(chat_completion):
    """  
    Extracts messages from the chat completion that have the role 'assistant' or 'tool'.  
  
    Args:  
        chat_completion: An object with a 'messages' attribute, which is a list of message objects.  
                         Each message object is expected to have a 'role' attribute.  
  
    Returns:  
        List of message objects where the 'role' is either 'assistant' or 'tool'.  
    """  
    messages = chat_completion.messages
    assistant_and_tool_messages = [message for message in messages if message.role in ['assistant', 'tool']]
    return assistant_and_tool_messages

def get_model_list():
    """
    Loads model_name from a YAML config file and returns a formatted list of model options.
    Ensures known models are included and current model is prioritized.
    """
    try:
        with open("./config.yml", 'r') as file:
            config_data = yaml.safe_load(file)

        # Validate config using Pydantic
        validated_config = ChatConfig(**config_data)

        # Set class attributes from validated config
        model_name = validated_config.model_name

        # Known recommended models
        known_models = {
            'databricks-claude-sonnet-4-6': 'Claude Sonnet 4.6'
        }

        if model_name in known_models:
            model_list = [{'label': label, 'value': value} for value, label in known_models.items()]
        else:
            # Ensure current model is included first, then known models
            model_list = [{'label': model_name, 'value': model_name}] + [
                {'label': label, 'value': value} for value, label in known_models.items()
            ]

        return model_list
    
    except FileNotFoundError:
        raise FileNotFoundError("Config file not found: ./config.yml")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")
    except ValueError as e:
        raise ValueError(f"Configuration validation error: {e}")


def update_genie_spaces():
    """
    Fetches the list of Genie spaces by making a REST API call using the token
    from the request headers. If the call is successful, extracts and returns
    the titles of the available spaces. Falls back to an empty list in case of
    errors or missing token.
    
    Returns:
        list: A list of space titles (str), or an empty list if unavailable.
    """
    # Fallback to direct API call
    cfg = Config()
    try:
        # Get the token from the request headers
        token = request.headers.get('X-Forwarded-Access-Token')
        if token:
            response = run_rest_api(
                server_hostname=cfg.host,
                token=token,
                api_version="2.0",
                api_command='genie/spaces',
                action_type='GET'
            )

            spaces_data = response.json()['spaces']
            genie_spaces = {space['title']: {'id': space['space_id'], 'description': space.get('description', '')} for space in spaces_data}

        else:
             genie_spaces = []
    except Exception as e:
        print(f"Error fetching Genie spaces: {e}")
        genie_spaces = []

    return  genie_spaces


def optimize_conversation_history(messages: List[Dict[str, str]], 
                                 max_tokens: int = 100000, 
                                 min_messages: int = 5,
                                 default_messages: int = 25,
                                 model: str = "databricks-claude-sonnet-4-6") -> List[Dict[str, str]]:
    """
    Optimizes conversation history to fit within token limits while preserving context.
    
    This function:
    1. Always keeps the system message (first message if role is 'system')
    2. By default keeps the last 25 messages (or as specified by default_messages)
    3. If token count exceeds max_tokens, dynamically reduces to keep at least min_messages
    4. Preserves conversation flow by keeping pairs of user/assistant messages when possible
    
    Args:
        messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'
        max_tokens (int): Maximum token limit (default: 100000 for Claude 3.7)
        min_messages (int): Minimum number of recent messages to keep (default: 5)
        default_messages (int): Default number of messages to keep if under token limit (default: 20)
        model (str): Model name to determine tokenizer (default: Claude 3.7)
    
    Returns:
        List[Dict[str, str]]: Optimized list of messages
    """
    if not messages:
        return []
    
    # Extract system message if present
    system_message = None
    other_messages = []
    
    for msg in messages:
        if msg.get('role') == 'system' and system_message is None:
            system_message = msg
        else:
            other_messages.append(msg)
    
    # Start with system message and the default number of recent messages
    keep_count = min(default_messages, len(other_messages))
    optimized_messages = []
    if system_message:
        optimized_messages.append(system_message)
    
    # Add the most recent messages up to keep_count
    optimized_messages.extend(other_messages[-keep_count:])
    
    # Check token count
    token_count = count_tokens(optimized_messages, model)
    
    # If we're under the limit, we're done
    if token_count <= max_tokens:
        return optimized_messages
    
    # If over the limit, reduce messages while keeping at least min_messages
    while token_count > max_tokens and keep_count > min_messages:
        keep_count -= 1
        
        # Rebuild the messages list
        optimized_messages = []
        if system_message:
            optimized_messages.append(system_message)
        optimized_messages.extend(other_messages[-keep_count:])
        
        # Recalculate token count
        token_count = count_tokens(optimized_messages, model)
    
    # If we still exceed the token limit with min_messages, we need to truncate content
    if token_count > max_tokens:
        # Start with just the system message and the absolute minimum messages
        optimized_messages = []
        if system_message:
            optimized_messages.append(system_message)
        optimized_messages.extend(other_messages[-min_messages:])
        
        # If still over limit, we could implement content truncation here
        # This is a simple implementation - you might want more sophisticated truncation
        token_count = count_tokens(optimized_messages, model)
        if token_count > max_tokens:
            print(f"Warning: Even with minimum messages, token count ({token_count}) exceeds limit ({max_tokens})")
    
    return optimized_messages

def count_tokens(messages: List[Dict[str, str]], model: str = "databricks-claude-sonnet-4-6") -> int:
    """
    Calculate the number of tokens in a list of messages.
    
    Args:
        messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'
        model (str): Model name to determine which tokenizer to use
    
    Returns:
        int: Total number of tokens
    """
    model = model.lower()
    
    # Using approximation based on Claude
    encoding_name = "cl100k_base"  
    tokens_per_message = 3
    tokens_per_name = 1
    
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    for message in messages:
        content = message.get('content', '')
        if content:
            total_tokens += tokens_per_message
            total_tokens += len(encoding.encode(content))
            total_tokens += tokens_per_name
    
    return total_tokens