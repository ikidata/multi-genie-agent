
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieAPI
import json
import time
from datetime import timedelta

def get_workspace_client():
    """
    Returns an instance of WorkspaceClient.
    """
    return WorkspaceClient()

def extract_column_values_string(response_obj):
    columns = [col.name for col in response_obj.statement_response.manifest.schema.columns]
    values = response_obj.statement_response.result.data_array[0]
    return ', '.join(f"{col}: {val}" for col, val in zip(columns, values))

def post_genie_and_wait(genie_space_id: str, prompt: str, w: object, timeout_min: int = 1) -> dict:
    """
    Starts a new conversation and waits for the Genie response.
    """
    response = w.genie.start_conversation_and_wait(
        space_id=genie_space_id,
        content=prompt,
        timeout=timedelta(minutes=timeout_min)
    )
    return response

def get_genie_message(genie_space_id: str,  w: object, conversation_id: str, message_id: str) -> dict:
    """
    Starts a new conversation and waits for the Genie response.
    """
    try:
        combinated_values = ""
        response = w.genie.get_message(
            space_id=genie_space_id,
            conversation_id=conversation_id,
            message_id = message_id
        )
        
        for attachment in response.attachments:
            attachment_id = attachment.attachment_id
            text = attachment.text.content if attachment.text is not None else ""
            
            if attachment.query is not None:
                query = attachment.query.query
                description = attachment.query.description

                query_values = w.genie.get_message_query_result_by_attachment(
                    space_id=genie_space_id,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    attachment_id=attachment_id)
                
                data_values = extract_column_values_string(query_values)
                combinated_values += combinated_values + f"\nDescription: {description}\nUsed query: {query}\nResults: {data_values}"
            else:
                combinated_values = text
            return combinated_values
    except Exception as e:
        return f"Error while fetching Genie's values: {e}"

def run_genie(genie_space_id: str, prompt: str, timeout_min: int = 1) -> str:
    '''
    Handles Genie API calls and returns the response.
    '''
    try:
        w = get_workspace_client()
        response = post_genie_and_wait(genie_space_id, prompt, w)  
        conversation_id = response.conversation_id
        message_id = response.id
        final_value = get_genie_message(genie_space_id, w, conversation_id, message_id)
    except Exception as e:
        final_value = f"Error during operating Genie: {e}"
    
    return final_value