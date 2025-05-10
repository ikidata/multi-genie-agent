import requests
import json
import yaml
import time
import os
from dbruntime.databricks_repl_context import get_context
from databricks.sdk import WorkspaceClient

def create_devops_connection(name: str, secret_scope: str, devops_token: str, devops_organization: str, devops_project: str): 

    # Retrieve the Databricks server hostname from the context  
    databricks_server_hostname = get_context().browserHostName
    databricks_server_hostname = f"https://{databricks_server_hostname}"
      
    # Retrieve the Databricks token from the context  
    databricks_token  = get_context().apiToken  

    w = WorkspaceClient()
    dbutils = w.dbutils
    devops_token = dbutils.secrets.get(secret_scope, devops_token)
    
    # Construct the payload for the connection  
    payload = {  
        "comment": "Genie-multi-agent DevOps PoC Connection",  
        "connection_type": "HTTP",  
        "name": name,  
        "options": {  
            "host": 'https://dev.azure.com',  
            "port": "443",  
            "base_path": f"/{devops_organization}/{devops_project}",  
            "bearer_token": devops_token,  
        },  
        "read_only": False  
    }  
  
    # Run the REST API command to create the connection  
    print(run_rest_api(  
        token=databricks_token,  
        server_hostname=databricks_server_hostname,  
        api_version="2.1",  
        api_command='/unity-catalog/connections',  
        action_type='POST',  
        payload=payload  
    ))
  

def create_genie_connection(name: str):  
  
    # Retrieve the Databricks server hostname from the context  
    databricks_server_hostname = get_context().browserHostName
    databricks_server_hostname = f"https://{databricks_server_hostname}"
      
    # Retrieve the Databricks token from the context  
    databricks_token  = get_context().apiToken 
      
    # Construct the payload for the connection  
    payload = {  
        "comment": "Genie-multi-agent PoC Connection",  
        "connection_type": "HTTP",  
        "name": name,  
        "options": {  
            "host": databricks_server_hostname,  
            "port": "443",  
            "base_path": "/api",  
            "bearer_token": databricks_token  
        },  
        "read_only": False  
    }  
      
    # Run the REST API command to create the connection  
    print(run_rest_api(  
        token=databricks_token,  
        server_hostname=databricks_server_hostname,  
        api_version="2.1",  
        api_command='/unity-catalog/connections',  
        action_type='POST',  
        payload=payload  
    ))


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

def create_config(secret_scope: str, databricks_token_secret_value: str, databricks_host_secret_value: str, databricks_genie_space_id_list: dict[str], genie_connection: str, devops_connection: str, devops_token_secret_value: str, devops_organization: str, devops_project: str):  
    tools = {}  
  
    for genie_space_id, description in databricks_genie_space_id_list.items():  
        tools[f"genie_{genie_space_id}"] = {  
            "type": "function",  
            "function": {  
                "name": f"genie_{genie_space_id}",  
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
        } 
    
    tools['get_documentation'] = {  
            "type": "function",  
            "function": {  
                "name": f"get_documentation",  
                "description": "Get internal documentation for data enrichment",  
                "parameters": {  
                    "type": "object",  
                    "properties": {  
                        "activation": {  
                            "type": "string",  
                            "description": "Yes or No"  
                        }  
                    },  
                    "required": ["activation"],  
                }
            }  
        } 
    
    if devops_connection != "":
        tools['create_update_devops_ticket'] = {  
                "type": "function",  
                "function": {  
                    "name": f"create_update_devops_ticket",  
                    "description": "A tool dedicated for Azure DevOps ticket creation and update",  
                    "parameters": {  
                        "type": "object",  
                        "properties": {  
                            "content": {  
                                "type": "string",  
                                "description": "write a content for DevOps ticket. Remove all formatting and bolding."  
                            }  
                        },  
                        "required": ["content"],  
                    }
                }  
            } 

    config = {  
        "system_prompt": "Only invoke a Genie space when the user prompt clearly aligns with the specific capability of that space (e.g., SQL generation, data exploration, reporting). Do not default to Genie unless the intent is explicit or strongly implied by the request. When invoking a Genie space, forward the prompt unchanged to preserve user intent. If processing a response from Genie, clean and format the output appropriately, and always indicate which Genie space was used. If the user prompt suggests a relevant task but Genie did not return a valid output, clearly state that no result was returned by Genie.",    
        "secret_scope": secret_scope,
        "databricks_token_secret_value": databricks_token_secret_value,
        "databricks_host_secret_value": databricks_host_secret_value,
        "genie_connection": genie_connection,
        "devops_connection": devops_connection,
        "devops_token_secret_value": devops_token_secret_value,
        "devops_organization": devops_organization,
        "devops_project": devops_project,
        "tools": tools
    }  
  
    with open("./apps/config.yml", "w") as file:  
        yaml.dump(config, file, default_flow_style=False) 


def check_deployment_status(token, server_hostname, app_name, payload, max_tries=25):  
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

def deploy_databricks_apps(name: str, model_name: str = "databricks-claude-3-7-sonnet") -> None:  
    max_tries = 25  
  
    # Load configuration data  
    with open("./apps/config.yml", 'r') as file:  
        config_data = yaml.safe_load(file)  
  
    # Construct the payload  
    payload = {  
        "description": "Multi-Genie-Agent PoC",  
        "name": name,  
        "resources": [  
            {  
                "description": "Serving endpoint for LLM model",  
                "name": "serving-endpoint",  
                "serving_endpoint": {  
                    "name": model_name,  
                    "permission": "CAN_QUERY"  
                }  
            }
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