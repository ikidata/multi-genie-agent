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


def create_databricks_token(server_hostname: str, client_id: str, client_secret: str) -> str:
    """
    Creates Databricks token with all-apis scope
    """
    # Construct the token endpoint URL 
    token_endpoint_url = f"{server_hostname}/oidc/v1/token" 

    # Prepare the data for the POST request 
    data = { 
        'grant_type': 'client_credentials', 
        'scope': 'all-apis' 
    } 

    # Make the POST request to get the access token 
    response = requests.post(token_endpoint_url, data=data, auth=(client_id, client_secret)) 

    # Check if the request was successful 
    assert response.status_code == 200, f"Failed to fetch Databricks access token \nError code: {response.status_code}\nError message: {response.json()}"

    # Parse the JSON response to get the access token 
    token = response.json().get('access_token') 
    return token


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

def convert_hostname_to_databricks_url(hostname: str) -> str:
    """
    Converts a Databricks Apps hostname to the standard workspace URL format.
    
    Example:
        'chatbotxd-15.14.azure.databricksapps.com' -> 'https://adb-15.14.azuredatabricks.net'
    """
    # Extract the numeric workspace ID from the hostname
    match = re.search(r'-(\d+\.\d+)\.azure\.databricksapps\.com', hostname)
    if not match:
        raise ValueError("Hostname does not match expected Databricks apps pattern.")

    workspace_id = match.group(1)
    return f"https://adb-{workspace_id}.azuredatabricks.net"