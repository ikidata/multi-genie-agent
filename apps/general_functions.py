import re
import requests
import json 

def call_chat_model(openai_client: any, model_name: str, messages: list, temperature: float = 0.3, max_tokens: int = 1000, **kwargs):
    """
    Calls the chat model and returns the response text or tool calls.

    Parameters:
        message (list): Message to send to the chat model.
        temperature (float, optional): Controls response randomness. Defaults to 0.3.
        max_tokens (int, optional): Maximum tokens for the response. Defaults to 750.
        **kwargs: Additional parameters for the chat model.

    Returns:
        str: Response from the chat model.
    """
    # Prepare arguments for the chat model
    chat_args = {
        "model": model_name,  
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs,  # Update with additional arguments
    }

    try:
        chat_completion = openai_client.chat.completions.create(**chat_args)
        return chat_completion  
    except Exception as e:
        print(f"Model endpoint calling error: {e}")
        return f"Model endpoint calling error: {e}"

def run_rest_api(server_hostname: str, token: str, api_version: str, api_command: str, action_type: str,  payload: dict = {}) -> str:
    """
    Run Databricks REST API endpoint dynamically
    """
    try:
        assert action_type in ['POST', 'GET'], f'Only POST and GET are supported but you used {action_type}'
        url = f"{server_hostname}/api/{api_version}{api_command}"
        headers = {'Authorization': 'Bearer %s' % token}
        session = requests.Session()
        
        resp = session.request(action_type, url, data=json.dumps(payload), verify=True, headers=headers)

        return resp
    except Exception as e:
        return e