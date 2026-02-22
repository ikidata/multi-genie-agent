import requests
import json
import yaml
import time
import os
from dbruntime.databricks_repl_context import get_context
from databricks.sdk import WorkspaceClient

def run_rest_api(token: str, server_hostname: str, api_version: str, api_command: str, action_type: str,  payload: dict = {}) -> str:
    """
    Executes a REST API call to the specified server using the provided parameters.


    This function constructs the API endpoint from the given server hostname, API version,  
    and API command. It then creates an authorization header with a bearer token and sends  
    an HTTP request (either GET or POST) along with any provided payload. The server's response,  
    expected to be in JSON format, is returned if the status code is 200.  
    
    Parameters:  
    token (str): Bearer token for authorization.  
    server_hostname (str): Hostname or URL of the REST API server.  
    api_version (str): API version to be included in the URL.  
    api_command (str): The specific API command or endpoint to call.  
    action_type (str): HTTP method for the request. Must be either 'GET' or 'POST'.  
    payload (dict, optional): Dictionary containing the request payload. Defaults to empty dict.  
        
    Returns:  
    str: The decoded JSON response if the request is successful.  
        If an error occurs, returns the Exception encountered.  
        
    Raises:  
    AssertionError: If an unsupported HTTP method is requested or if the response status code is not 200.  
        
    Note:  
    This function uses a persistent session (via requests.Session()) and verifies the server's SSL certificate.  
    """  
    try:
        assert action_type in ['POST', 'GET'], f'Only POST and GET are supported but you used {action_type}'
        url = f"{server_hostname}/api/{api_version}{api_command}"
        headers = {'Authorization': 'Bearer %s' % token}
        session = requests.Session()

        resp = session.request(action_type, url, data=json.dumps(payload), verify=True, headers=headers)
        assert resp.status_code == 200, f"Running REST API has failed with an error message: {resp.json()}"
        result = resp.json()
        return result
    except Exception as e:
        return e

def get_llm_model_apps_configs(model_name: str) -> list[dict]:
    """
    Generate a list of LLM model configurations for serving endpoints based on the provided model name.
    
    This function creates a prioritized list of serving endpoint configurations. If the provided
    model name is one of the default recommended models, it will be included in the list.
    Otherwise, the provided model will be added as the primary endpoint, followed by the
    default recommended models.
    
    Args:
        model_name (str): The name of the LLM model to use
        
    Returns:
        list[dict]: A list of serving endpoint configurations
    """
    # Known recommended models
    default_models = ['databricks-claude-sonnet-4-6']
    
    # Default model configurations
    claude_config = {  
        "description": "Serving endpoint for LLM model Claude 4.6. Sonnet",  
        "name": "serving-endpoint",  
        "serving_endpoint": {  
            "name": 'databricks-claude-sonnet-4-6',  
            "permission": "CAN_QUERY"  
        }  
    }
    
    
    if model_name in default_models:
        # If using a default model, just return the default configurations
        return [claude_config]
    else:
        # If using a custom model, add it as the primary endpoint followed by defaults
        custom_config = {  
            "description": f"Serving endpoint for LLM model {model_name}",  
            "name": "serving-endpoint",  
            "serving_endpoint": {  
                "name": model_name,  
                "permission": "CAN_QUERY"  
            }  
        }
        
        # Ensure the custom model is the first in the list
        return [custom_config, claude_config]

def create_config(model_name: str):  
    
    with open('./apps/config.yml', "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open('./apps/config.yml', "w", encoding="utf-8") as f:
        for line in lines:
            if line.strip().startswith("model_name:"):
                # preserve original indentation
                indent = line[:line.find("model_name:")]
                f.write(f"{indent}model_name: {model_name}\n")
            else:
                f.write(line)
        

def check_deployment_status(token, server_hostname, app_name, payload, max_tries=25):  
    """
    Polls the deployment status of an application until it becomes active or the retry limit is reached.

    Args:
        token (str): Authentication token for the API.
        server_hostname (str): Hostname of the server where the app is being deployed.
        app_name (str): Name of the application being deployed.
        payload (dict): The payload to send with the API request.
        max_tries (int, optional): Maximum number of polling attempts. Defaults to 25.

    Returns:
        dict or str: Deployment result if successful, or an error message string if the deployment doesn't complete in time.
    """
    tries = 0  
      
    while tries < max_tries:  
        deployment_results = run_rest_api(  
            token=token,  
            server_hostname=server_hostname,  
            api_version="2.0",  
            api_command=f'/apps/{app_name}',  
            action_type='GET',  
            payload=payload  
        )  
          
        deployment_status = deployment_results['compute_status']['state']  
        if deployment_status == 'ACTIVE':  
            print(f"Deployment of {app_name} completed successfully.")  
            return deployment_results  
        else:  
            print(f"Deploying {app_name}, please be patient. Waiting 15 seconds before retrying...")  
            time.sleep(15)  
            tries += 1  
      
    print(f"Deployment of {app_name} did not complete after {max_tries} attempts.")  
    return f"Deployment of {app_name} did not complete after {max_tries} attempts."

def deploy_databricks_apps(name: str, model_name: str = "databricks-claude-sonnet-4-6") -> None:  
    """
    Deploys a Databricks application with the specified model and monitors its deployment status.

    Args:
        name (str): Name of the application to be deployed.
        model_name (str, optional): Name of the model serving endpoint to attach. Defaults to Claude 3 Sonnet.

    Returns:
        None
    """
    max_tries = 25  
  
    # Load configuration data  
    with open("./apps/config.yml", 'r') as file:  
        config_data = yaml.safe_load(file)  

    # Construct the payload  
    payload = {  
        "description": "Agentic Multi-Genie-Agent Solution",  
        "name": name,  
        "resources": get_llm_model_apps_configs(model_name),  
        "user_api_scopes": [
            "dashboards.genie"
        ]  
    }  
  
    # Retrieve the Databricks server hostname from the context  
    databricks_server_hostname = get_context().browserHostName
    databricks_server_hostname = f"https://{databricks_server_hostname}"
      
    # Retrieve the Databricks token from the context  
    databricks_token  = get_context().apiToken 
  
    # Deploy the application  
    response = run_rest_api(  
        token=databricks_token,  
        server_hostname=databricks_server_hostname,  
        api_version="2.0",  
        api_command='/apps',  
        action_type='POST',  
        payload=payload  
    )  
    
    deployment_id = response['id']  
    print(f"deployment_id is {deployment_id}")
  
    # Check deployment status 
    deployment_successful = check_deployment_status(  
        token=databricks_token,  
        server_hostname=databricks_server_hostname,  
        app_name=name,  
        payload=payload  
    )

    databricks_sp_app_id = deployment_successful['service_principal_client_id']  
  
    # Update payload for deployment  
    deployment_payload = {  
        "deployment_id": deployment_id,  
        "mode": "SNAPSHOT",  
        "source_code_path": f"{os.getcwd()}/apps"
    }  

    # Deploy the application  
    run_rest_api(  
        token=databricks_token,  
        server_hostname=databricks_server_hostname,  
        api_version="2.0",  
        api_command=f'/apps/{name}/deployments',  
        action_type='POST',  
        payload=deployment_payload 
    ) 
    print("##################################################")
    print("")
    print("Congratulations, App deployment is ready - time to have some fun!")
    print("")
    print(f"Remember to grant access to Genie Spaces for the next App ID: {databricks_sp_app_id}")  