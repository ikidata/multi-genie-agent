from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ExternalFunctionRequestHttpMethod
import json
import time

def get_workspace_client():
    """
    Returns an instance of WorkspaceClient.
    """
    return WorkspaceClient()

def post_genie(genie_space_id: str, genie_conn: str, prompt: str) -> dict:
    """
    Starts a new conversation with Genie by sending a prompt.

    Args:  
        prompt (str): The prompt text to send to Genie.  
    
    Returns:  
        dict: The HTTP response from the POST request.  
    """  
    client = get_workspace_client()  
    response = client.serving_endpoints.http_request(  
        conn=genie_conn,  
        method=ExternalFunctionRequestHttpMethod.POST,  
        path=f"/2.0/genie/spaces/{genie_space_id}/start-conversation",  
        json={"content": prompt}  
    )  
    return response  
 
def get_genie_query_results(genie_space_id: str, genie_conn: str, conversation_id: str, message_id: str) -> dict:
    """
    Retrieves the current status and any early results for a Genie conversation.


    Args:  
        conversation_id (str): The conversation ID received after the prompt was sent.  
        message_id (str): The message ID received after the prompt was sent.  
    
    Returns:  
        dict: The parsed JSON response from the GET request.  
    """  
    client = get_workspace_client()  
    response = client.serving_endpoints.http_request(  
        conn=genie_conn,  
        method=ExternalFunctionRequestHttpMethod.GET,  
        path=f"/2.0/genie/spaces/{genie_space_id}/conversations/{conversation_id}/messages/{message_id}"  
    )  
    return response.json()  
 
def get_genie_query_attachment_results(genie_space_id: str, genie_conn: str, conversation_id: str, message_id: str, attachment_id: str) -> str:
    """
    Retrieves query results from a specific attachment from Genie.

    Args:  
        conversation_id (str): The conversation ID.  
        message_id (str): The message ID.  
        attachment_id (str): The attachment ID from which to fetch the query result.  
    
    Returns:  
        str: A string representation of the query result.  
    """  
    client = get_workspace_client()  
    response = client.serving_endpoints.http_request(  
        conn=genie_conn,  
        method=ExternalFunctionRequestHttpMethod.GET,  
        path=f"/2.0/genie/spaces/{genie_space_id}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result"  
    )  
    data_array = response.json()['statement_response']['result']['data_array']  
    return str(data_array)  
 
def run_genie(genie_space_id: str, genie_conn: str, prompt: str, wait_seconds: int = 1, max_retries: int =  30) -> str:
    """
    Main routine to:
    1. Post a prompt to Genie.
    2. Poll for conversation status until completed (or maximum retries reached).
    3. Process the response from Genie.

    Args:  
        genie_space_id (str): The unique identifier for the Genie space.  
        genie_conn (str): The connection string or credentials for accessing Genie.  
        prompt (str): The prompt/question for Genie.  
        wait_seconds (int, optional): The number of seconds to wait between polling attempts. Defaults to 1.  
        max_retries (int, optional): The maximum number of polling attempts before giving up. Defaults to 15.  
  
    Returns:  
        str: The processed response from Genie.  
    """  
    response = post_genie(genie_space_id, genie_conn, prompt)  
    
    if response.status_code == 200:  
        try:  
            raw_post_value = json.loads(response.text)  
        except json.JSONDecodeError as exc:  
            print("JSON decode error on POST response:", exc)  
            return "Genie JSON decode error on POST response:", exc

        conversation_id = raw_post_value.get('conversation_id')  
        message_id = raw_post_value.get('message_id')  
        if not conversation_id or not message_id:  
            print("Missing conversation_id or message_id in the response.")  
            return "Genie Missing conversation_id or message_id in the response."

        status = 'IN_PROGRESS'  
        current_try = 0  
        raw_get_value = {}  

        while status != 'COMPLETED' and current_try < max_retries:  
            raw_get_value = get_genie_query_results(genie_space_id, genie_conn, conversation_id, message_id)  
            status = raw_get_value.get('status', 'UNKNOWN')  
            current_try += 1  
            print("Waiting for completion... (try", current_try, "of", max_retries,")")  
            if status != 'COMPLETED':  
                time.sleep(wait_seconds)  

        if status != 'COMPLETED':  
            print( f"Genie query did not complete after {max_retries} retries.")  
            return f"Genie query did not complete after {max_retries} retries."

        attachments = raw_get_value.get('attachments', [])  
        if not attachments:  
            print("No attachments found in the Genie response.")  
            return "No attachments found in the Genie response."

        attachment_value = attachments[0]  
        attachment_id = attachment_value.get('attachment_id')  
        if not attachment_id:  
            print("No attachment_id found in the first attachment.")  
            return "No attachment_id found in the first Genie attachment."

        if 'text' in attachment_value:  
            text_content = attachment_value['text'].get('content', '')  
            print(text_content)  
            return text_content
        
        elif 'query' in attachment_value:  
            query_description = attachment_value['query'].get('description', '')  
            try:  
                query_results = get_genie_query_attachment_results(genie_space_id, genie_conn, conversation_id, message_id, attachment_id)  
                final_value = "Text: " + query_description + "\nQuery: " + query_results  
            except Exception as e:  
                print("Error retrieving query attachment results:", e)  
                final_value = query_description  
            print(f"Final Genie value: {final_value}")  
            return final_value
        else:  
            print("Failed to decode Genie results from the attachment.")  
    else:  
        try:  
            error_message = response.json()  
        except Exception:  
            error_message = response.text  
        print("Error with Genie:", error_message)  
        return f"Error with Genie: {error_message}"