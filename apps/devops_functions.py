from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ExternalFunctionRequestHttpMethod
import json 
from datetime import datetime

def fetch_active_epic(connection: str) -> list:
    """  
    Fetches a list of active epic work item IDs from Azure DevOps that contain a specific title and are not marked as 'Done'.  
  
    Returns:  
        list: A list of work item IDs that match the query criteria.  
    """  
      
    # Define the Work Item Query Language (WIQL) query to fetch the required work items. Here it's locked by hard coding due to demo purposes.
    wiql_query = f"""
            SELECT 
                [System.Id]
            FROM 
                workitems
            WHERE 
                [System.Title] CONTAINS 'Agent is your new colleague'
            AND 
                [System.State] <> 'Done'
            ORDER BY 
                [System.ChangedDate] DESC
            """

    # Send the WIQL query to the Azure DevOps API to execute the query 
    response = WorkspaceClient().serving_endpoints.http_request(
        conn=connection,
        method=ExternalFunctionRequestHttpMethod.POST,
        path=f"/_apis/wit/wiql?api-version=7.0",
        json={
                "query": wiql_query 
            }
    )

    # Extract the work item IDs from the response  
    work_item_ids = [item['id'] for item in response.json()['workItems']]  
      
    # Return the list of work item IDs  
    return work_item_ids  

def create_devops_ticket(content: str, connection: str) -> str:
    """  
    Creates or updates a DevOps ticket in Azure DevOps.  
  
    Args:  
        content (str): The content to be included in the DevOps ticket description.  

    Returns:  
        str: A message indicating the result of the DevOps ticket creation or update.  
    """  
    # Get the current date in YYYY-MM-DD format  
    current_date = datetime.now().strftime("%Y-%m-%d")  
      
    # Fetch the list of active epics  
    current_epic = fetch_active_epic(connection)  
      
    # Determine the URL and HTTP method based on whether an active epic exists  
    if len(current_epic) != 0:
        url = f"/_apis/wit/workitems/{current_epic[0]}?api-version=7.1"
        method = ExternalFunctionRequestHttpMethod.PATCH
        return_value = "DevOps Epic creation was successful"
    else:
        url = '/_apis/wit/workitems/$Epic?api-version=7.1'
        method = ExternalFunctionRequestHttpMethod.POST
        return_value = "DevOps Epic updating was successful"

    fields = [  
        {  
            "op": "add",  
            "path": "/fields/System.Title",  
            "value": f"Agent is your new colleague"          # Title is locked for demo purpose
        },
        {  
            "op": "add",  
            "path": "/fields/System.Description",  
            "value": f"""<b>{content}:</b>
                        <br/><br/>
                        <br/><br/>
                        <br/><br/>
                        <img src='https://cdn.cookielaw.org/logos/29b588c5-ce77-40e2-8f89-41c4fa03c155/bc546ffe-d1b7-43af-9c0b-9fcf4b9f6e58/1e538bec-8640-4ae9-a0ca-44240b0c1a20/databricks-logo.png' alt='Databricks logo' width='150'/>
                    """                                      # Here you can choose a static or dynamic pictures, hyperlinks etc.
        },  
        {  
            "op": "add",  
            "path": "/fields/System.AssignedTo",  
            "value": ""                                      # Here you can choose a static or dynamic value - currently empty.
        },  
        {  
            "op": "add",  
            "path": "/fields/System.Tags",  
            "value": "Databricks, GenAI Agents, Automation"  # Replace with your actual tags  
        }
    ]  
    # Send the request to Azure DevOps API to create or update the ticket 
    response = WorkspaceClient().serving_endpoints.http_request(
        conn=connection,
        method=method, 
        path=url,
        headers={"Content-Type": "application/json-patch+json"},
        json=fields
    )
    
    # Check the response status  
    if response.status_code == 200 or response.status_code == 201:
        return return_value
    else:
        return f"DevOps Epic creation failed with status code {response.status_code} and error message {response.json()}"