from typing import Dict, Tuple
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import json
import time
import mlflow
from mlflow.entities import SpanType

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config


class ToolException(Exception):
    """Exception raised for Genie tool validation errors."""
    pass

class GenieRunInput(BaseModel):
    """Pydantic model for validating run_genie inputs."""
    genie_space_id: str = Field(..., description="The ID of the Genie space to interact with")
    prompt: str = Field(..., description="The prompt/question to send to the Genie")
    obo_token: str = Field(..., description="A dash header OBO token for Genie authentication")
    conversation_id: Optional[str] = Field(None, description="Genie space conversation ID (optional)")
    sleeper_time: float = Field(0.7, description="Time to wait between retries when polling for message status (seconds)")
    max_retries: int = Field(45, description="Maximum number of retries before giving up on the message status")

    @field_validator('genie_space_id')
    @classmethod
    def validate_genie_space_id(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ToolException("genie_space_id must be a non-empty string")
        return v

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ToolException("prompt must be a non-empty string")
        return v

    @field_validator('obo_token')
    @classmethod
    def validate_obo_token(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ToolException("obo_token must be a non-empty string")
        return v
    
    @field_validator('conversation_id')
    @classmethod
    def validate_conversation_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not isinstance(v, str):
            raise ToolException("conversation_id must be a string or None")
        return v

    @field_validator('sleeper_time')
    @classmethod
    def validate_sleeper_time(cls, v: float) -> float:
        if v <= 0:
            raise ToolException("sleeper_time must be positive")
        return v

    @field_validator('max_retries')
    @classmethod
    def validate_max_retries(cls, v: int) -> int:
        if v <= 0:
            raise ToolException("max_retries must be positive")
        return v

def get_genie_name_by_id(genie_spaces: dict[str, dict[str, str]], genie_id: str):
    """
    Returns the name of a Genie space based on its ID.
    
    Args:
        genie_spaces (dict): Dictionary of Genie spaces
        genie_id (str): The ID to look up
        
    Returns:
        str or None: The name of the Genie space if found, None otherwise
    """
    for name, details in genie_spaces.items():
        if details['id'] == genie_id:
            return name
    return None
    

def extract_column_values_string(response_obj):
    """
    Extracts and formats up to 50 rows of column-value pairs from a response object.

    Args:
        response_obj: An object containing a SQL or query result with schema and data.

    Returns:
        List[str]: A list of comma-separated strings of column-value pairs for each row,
                   e.g. ["col1: val1, col2: val2", "col1: val3, col2: val4"].
    """
    # Extract column names from the response schema
    columns = [col.name for col in response_obj.statement_response.manifest.schema.columns]

    # Extract up to 50 rows of data
    data_rows = response_obj.statement_response.result.data_array[:50]

    # Format each row as a string of column-value pairs
    formatted_rows = [
        ', '.join(f"{col}: {val}" for col, val in zip(columns, row))
        for row in data_rows
    ]

    return formatted_rows


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


def create_message_genie(genie_space_id: str, prompt: str, conversation_id: str, w: object) -> dict:
    """
    Create a message to existing Genie Space.

    Args:
        genie_space_id (str): The ID of the Genie space where the conversation should occur.
        prompt (str): The user input or message to send to the Genie.
        converstaaion_id (str): The ID of the Genie conversation.
        w (object): A Databricks WorkspaceClient object with access to the Genie API.

    Returns:
        dict: Result message from Genie
    """
    response = w.genie.create_message(
        space_id=genie_space_id,
        content=prompt,
        conversation_id=conversation_id
    )
    return response


def get_genie_message(genie_space_id: str,  w: object, conversation_id: str, message_id: str, sleeper_time: float = 0.7, max_retries: int =  60) -> dict:
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
            
            # Returning error messages correctly
            if response_status == 'FAILED':
                print("Genie query failed...") 
                return response.error.error
            current_try += 1
            time.sleep(0.7)

        if response_status != 'COMPLETED':  
            print( f"Genie query did not complete after {max_retries} retries.")  
            return f"Genie query did not complete after {max_retries} retries."
            
        # Iterate over all attachments to extract data or fallback text
        for attachment in response.attachments:
            # Check if the attachment object has a valid ID or other identifying attribute  
            if not hasattr(attachment, 'attachment_id') or attachment.attachment_id is None:  
                print("Genie query didn't provide any valid attachments, so probably it failed...")
                return "Genie query didn't provide any valid attachments, so probably it failed..."

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

#@mlflow.trace(name="run_genie", span_type=SpanType.TOOL)
def run_genie(genie_space_id: str, prompt: str, obo_token: str, conversation_id: str = None, sleeper_time: float = 0.7, max_retries: int = 45) -> Tuple[str, str]:
    """
    Executes a full Genie interaction by posting a prompt and retrieving the final message output.

    Args:
        genie_space_id (str): The ID of the Genie space to interact with.
        prompt (str): The prompt/question to send to the Genie.
        obo_token (str): The user token fetched from Dash headers which is used to authenticate go Genie Spaces
        conversation_id (str): The ID of the existing conversation to continue, or None to start a new one.
        sleeper_time (float): The time to wait between retries when polling for the message status (seconds).
        max_retries (int): The maximum number of retries before giving up on the message status.

    Returns:
        str: The final text or query-based response from Genie. If an error occurs, an error message is returned.
    """
    try:
        # Validate inputs using Pydantic
        input_data = GenieRunInput(
            genie_space_id=genie_space_id,
            prompt=prompt,
            obo_token=obo_token,
            conversation_id=conversation_id,
            sleeper_time=sleeper_time,
            max_retries=max_retries
        )
        
        user_config = Config(
            host=Config().host,   
            token=obo_token,
            # Disable all OAuth settings
            client_id=None,
            client_secret=None,
            tenant_id=None,
            # Ensure no environment variables are used
            use_azure_cli=False,
            auth_type="pat"  # Explicitly specify PAT authentication
        )

        # Create PAT user auth SDK client
        we = WorkspaceClient(config=user_config) 

        # Using existing conversation if it exists
        if conversation_id:
            response = create_message_genie(input_data.genie_space_id, input_data.prompt, input_data.conversation_id, we)

        else:
            # Start a conversation and wait for Genie to respond
            response = post_genie(input_data.genie_space_id, input_data.prompt, we) 

            # Extract conversation and message IDs for follow-up query 
            conversation_id = response.conversation_id

        # Extract message IDs for follow-up query 
        message_id = response.message_id

        # Retrieve and return the processed response and convert it to string
        final_value = str(get_genie_message(
            input_data.genie_space_id, 
            we, 
            conversation_id, 
            message_id, 
            input_data.sleeper_time, 
            input_data.max_retries
        ))

        # Ensure the conversation ID exists
        conversation_id = conversation_id or None

    except ToolException as e:
        final_value = f"Genie tool validation error: {str(e)}"
    except Exception as e:
        final_value = f"Error during operating Genie: {e}"
    
    return final_value, conversation_id