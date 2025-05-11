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
    """
    Extracts and formats the first row of column-value pairs from a response object.

    Args:
        response_obj: An object containing a SQL or query result with schema and data.

    Returns:
        str: A comma-separated string of column-value pairs, e.g. "col1: val1, col2: val2".
    """
    # Extract column names from the response schema
    columns = [col.name for col in response_obj.statement_response.manifest.schema.columns]

    # Get the first row of data values
    values = response_obj.statement_response.result.data_array[0]

    # Pair columns with their corresponding values and return as formatted string
    return ', '.join(f"{col}: {val}" for col, val in zip(columns, values))

def post_genie(genie_space_id: str, prompt: str, w: object) -> dict:
    """
    Starts a new Genie conversation in a specified space.

    Args:
        genie_space_id (str): The ID of the Genie space where the conversation should occur.
        prompt (str): The user input or message to send to the Genie.
        w (object): A Databricks WorkspaceClient object with access to the Genie API.

    Returns:
        dict: The full Genie response object containing the output of the conversation.
    """
    response = w.genie.start_conversation(
        space_id=genie_space_id,
        content=prompt
    )
    return response

def get_genie_message(genie_space_id: str,  w: object, conversation_id: str, message_id: str, sleeper_time: float = 0.7, max_retries: int =  45) -> dict:
    """
    Retrieves a specific Genie message and its associated query results (if any) from a conversation.

    Args:
        genie_space_id (str): The Genie space ID where the conversation is located.
        w (object): A Databricks WorkspaceClient instance with Genie API access.
        conversation_id (str): The ID of the Genie conversation.
        message_id (str): The ID of the specific message within the conversation to fetch.
        sleeper_time (float): The time to wait between retries when polling for the message status (seconds).
        max_retries (int): The maximum number of retries before giving up on the message status.

    Returns:
        str: A formatted string combining any text, query, and query result found in the message attachments.
             If an error occurs, a string describing the error is returned.
    """
    try:
        combinated_values = ""
        response_status = ""
        current_try = 1
        # Looping until response is received
        while response_status != "COMPLETED" and current_try <= max_retries:
            response = w.genie.get_message(
                space_id=genie_space_id,
                conversation_id=conversation_id,
                message_id = message_id
            )
            print("⏳ Waiting for completion... (try", current_try, "of", max_retries,")") 
            response_status = response.status.value
            current_try += 1
            time.sleep(0.7)

        if response_status != 'COMPLETED':  
            print( f"Genie query did not complete after {max_retries} retries.")  
            return f"Genie query did not complete after {max_retries} retries."
            
        # Iterate over all attachments to extract data or fallback text
        for attachment in response.attachments:
            attachment_id = attachment.attachment_id
            text = attachment.text.content if attachment.text is not None else ""
            
            if attachment.query is not None:
                query = attachment.query.query
                description = attachment.query.description

                # Fetch actual query result data
                query_values = w.genie.get_message_query_result_by_attachment(
                    space_id=genie_space_id,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    attachment_id=attachment_id)
                
                # Format query result as string
                data_values = extract_column_values_string(query_values)
                combinated_values += combinated_values + f"\nDescription: {description}\nUsed query: {query}\nResults: {data_values}"
            else:
                # If no query, just return the text content
                combinated_values = text
            return combinated_values
    except Exception as e:
        return f"Error while fetching Genie's values: {e}"

def run_genie(genie_space_id: str, prompt: str, sleeper_time: float = 0.7, max_retries: int =  30) -> str:
    """
    Executes a full Genie interaction by posting a prompt and retrieving the final message output.

    Args:
        genie_space_id (str): The ID of the Genie space to interact with.
        prompt (str): The prompt/question to send to the Genie.
        sleeper_time (float): The time to wait between retries when polling for the message status (seconds).
        max_retries (int): The maximum number of retries before giving up on the message status.

    Returns:
        str: The final text or query-based response from Genie. If an error occurs, an error message is returned.
    """
    try:
        # Initialize the Workspace client
        w = get_workspace_client()
        
        # Start a conversation and wait for Genie to respond
        response = post_genie(genie_space_id, prompt, w) 

        # Extract conversation and message IDs for follow-up query 
        conversation_id = response.conversation_id
        message_id = response.message_id

        # Retrieve and return the processed response
        final_value = get_genie_message(genie_space_id, w, conversation_id, message_id, sleeper_time, max_retries)
    except Exception as e:
        final_value = f"Error during operating Genie: {e}"
    
    return final_value