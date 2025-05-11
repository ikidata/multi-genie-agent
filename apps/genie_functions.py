
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

def post_genie_and_wait(genie_space_id: str, prompt: str, w: object, timeout_min: int = 1) -> dict:
    """
    Starts a new Genie conversation in a specified space and waits for the response.

    Args:
        genie_space_id (str): The ID of the Genie space where the conversation should occur.
        prompt (str): The user input or message to send to the Genie.
        w (object): A Databricks WorkspaceClient object with access to the Genie API.
        timeout_min (int, optional): Maximum time to wait for a response in minutes. Default is 1 minute.

    Returns:
        dict: The full Genie response object containing the output of the conversation.
    """
    response = w.genie.start_conversation_and_wait(
        space_id=genie_space_id,
        content=prompt,
        timeout=timedelta(minutes=timeout_min)
    )
    return response

def get_genie_message(genie_space_id: str,  w: object, conversation_id: str, message_id: str) -> dict:
    """
    Retrieves a specific Genie message and its associated query results (if any) from a conversation.

    Args:
        genie_space_id (str): The Genie space ID where the conversation is located.
        w (object): A Databricks WorkspaceClient instance with Genie API access.
        conversation_id (str): The ID of the Genie conversation.
        message_id (str): The ID of the specific message within the conversation to fetch.

    Returns:
        str: A formatted string combining any text, query, and query result found in the message attachments.
             If an error occurs, a string describing the error is returned.
    """
    try:
        combinated_values = ""
        response = w.genie.get_message(
            space_id=genie_space_id,
            conversation_id=conversation_id,
            message_id = message_id
        )
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

def run_genie(genie_space_id: str, prompt: str, timeout_min: int = 1) -> str:
    """
    Executes a full Genie interaction by posting a prompt and retrieving the final message output.

    Args:
        genie_space_id (str): The ID of the Genie space to interact with.
        prompt (str): The prompt/question to send to the Genie.
        timeout_min (int, optional): Time in minutes to wait for a Genie response. Defaults to 1.

    Returns:
        str: The final text or query-based response from Genie. If an error occurs, an error message is returned.
    """
    try:
        # Initialize the Workspace client
        w = get_workspace_client()
        
        # Start a conversation and wait for Genie to respond
        response = post_genie_and_wait(genie_space_id, prompt, w) 

        # Extract conversation and message IDs for follow-up query 
        conversation_id = response.conversation_id
        message_id = response.id

        # Retrieve and return the processed response
        final_value = get_genie_message(genie_space_id, w, conversation_id, message_id)
    except Exception as e:
        final_value = f"Error during operating Genie: {e}"
    
    return final_value