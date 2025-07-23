from databricks.sdk import WorkspaceClient
from mlflow import tracing
from mlflow.tracing.destination import MlflowExperiment
from mlflow.entities import SpanType
from mlflow.pyfunc.model import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext)
import mlflow
import json

from openai import OpenAI
from typing import List, Generator, Any, Optional, Dict
import os
import re 
from uuid import uuid4

from general_functions import call_chat_model, create_tool_calls_output, create_tools_from_genie_spaces
from genie_functions import run_genie

### Set up MLflow tracing
#mlflow.openai.autolog(log_traces=False)
# experiment_name = "/multi-genie-solution"  # You can change mlflow tracing destination and experiment name as well
# mlflow.set_experiment(experiment_name)
# experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
# mlflow.tracing.set_destination(MlflowExperiment(experiment_id=experiment_id))

#### Agent class
class SimpleChatBotAgent(ChatAgent):
    def __init__(self,         
        model_name: str,
        max_tokens: int,
        obo_token: str = None,
        genie_spaces: dict[str, dict[str, str]] = None):

        """
        Set parameters for the model.
        
        Args:
            model: The model identifier to use
            max_tokens: Maximum number of tokens to generate
            obo_token: On-behalf-of token (optional)
            genie_spaces: Dictionary of genie spaces for tool creation (optional)
        """

        self.w_system = WorkspaceClient()                                         # Create system SDK connection
        self.openai_client = self.w_system.serving_endpoints.get_open_ai_client() # Create OpenAI Client - gief Async client...

        # Adding place holder values for the agent
        self.genie_conversation_dict = {}

        self.obo_token = obo_token or ''
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.genie_spaces = genie_spaces

        if self.genie_spaces:
            self.tools = create_tools_from_genie_spaces(self.genie_spaces)
        else:
            self.tools = []

    #@mlflow.trace(name="stringify_tool_call", span_type=SpanType.CHAIN)    
    def stringify_tool_call(self, response: object) -> dict:
        """
        Extracts and formats information from a tool call response.

        Parameters:
            response (object): The response object containing choices, messages, and tool call details.

        Returns:
            dict: A dictionary with the role, content, and tool call details from the response.
        """

        try:
            # Extract message details
            value = response.choices[0].message
            if not value.content:
                content = "-"
            else:
                content= value.content
                    
            # Construct payload
            payload = {
                "role": value.role,
                "content": content,
                "name": None,  # No name provided, use None
                "id": response.id,  # New id with 'run-' prefix
                "tool_calls": create_tool_calls_output(value),
                "tool_call_id": None,  # No tool_call_id in the input, set to None
                "attachments": None  # No attachments in the input, set to None
            } 

            return ChatAgentMessage(**payload) 
        
        except (AttributeError, IndexError) as e:
            raise ValueError(f"Invalid response format: {e}")
         

    #@mlflow.trace(name="process_tool_calls", span_type=SpanType.CHAIN)        
    def process_tool_calls(self, response: object) -> list:
        """
        Processes a list of tool calls and maps them to appropriate functions for execution.

        Parameters:
            tool_calls (object): A list of tool call objects containing function details.

        Returns:
            list: A list of payload dictionaries representing the processed tool calls.
        """
        # Create placeholder messages list
        new_messages = []

        # Start processing tool call one by one
        tool_calls = response.choices[0].message.tool_calls

        for tool_call in tool_calls:
            try:
                # Extract function name and arguments
                function_name = tool_call.function.name
                function_arguments = json.loads(tool_call.function.arguments)

                # Get the mapped function and execute genie
                if "genie_" in function_name:

                    # Fetching conversation_id
                    conversation_id = self.genie_conversation_dict.get(function_name, '')

                    tool_prompt = function_arguments['prompt']
                    genie_space_id = function_name.split('_')[1]

                    function_output, conversation_id = run_genie(genie_space_id = genie_space_id, obo_token = self.obo_token, conversation_id = conversation_id, prompt = tool_prompt)

                    # Updating Genie conversation dict 
                    if conversation_id:
                        self.genie_conversation_dict[function_name] = conversation_id

                # Parse other function arguments
                else:
                    function_arguments = json.loads(tool_call.function.arguments)
                    # Execute the function
                    function_output = function(**function_arguments)

                # Add validator that function_output is ALWAYS a string
                if not isinstance(function_output, str):
                    raise ValueError(f"Function '{function_name}' did not return a string.")
                
                # Create the payload
                payload = {
                    "role": "tool",
                    "content": function_output,
                    "name": function_name,
                    "tool_call_id": tool_call.id,
                    "id": str(uuid4())
                }

                # Append the processed payload
                new_messages.append(ChatAgentMessage(**payload))
            except Exception as e:
                raise RuntimeError(f"Error processing tool call {tool_call.id}: {e}")
        return new_messages


    def agent_tool_calling(self, messages, max_node_count: int = 7, node_count: int = 1):
        """  
        Runs an iterative agent loop that calls a chat model with tools, processes responses,  
        and yields messages until a stopping condition is met or a maximum number of iterations is reached.  
    
        Args:  
            messages (list): List of message objects to send as context to the chat model.  
            max_node_count (int, optional): Maximum number of iterations (nodes) to process. Defaults to 7.  
            node_count (int, optional): Starting node count (iteration counter). Defaults to 1.  
    
        Yields:  
            ChatAgentMessage or str: Yields assistant or tool messages generated by the agent, or a warning string if token limit is reached.  
    
        Raises:  
            Exception: Any exception that occurs during execution.  
        """  

        try:
            result = call_chat_model(self.openai_client, self.model_name, messages, self.max_tokens, tools = self.tools)
            if result.choices[0].finish_reason == 'length':
                yield "You are running out of tokens. Please reduce the number of tokens or increase the node count."
     
            if result.choices[0].finish_reason == 'stop':
                yield ChatAgentMessage(**result.choices[0].message.to_dict(), id=result.id)

            else:
                # Fetching the assistant message
                assistant_message = self.stringify_tool_call(result)
                messages.append(assistant_message)
                yield assistant_message

                # Processing tool_calls
                for temp_message in self.process_tool_calls(result):  
                    messages.append(temp_message)
                    yield temp_message

                # Calling chat model again
                result = call_chat_model(self.openai_client, self.model_name, messages, self.max_tokens)
                yield ChatAgentMessage(**result.choices[0].message.to_dict(), id=result.id)

        except Exception as e:
            error_message = {
                "role": "assistant",
                "content": f"Error occurred: {e}"
            }
            yield ChatAgentMessage(**error_message, id=str(uuid4()))

    ### ChatAgent requires predict function even when it's not being used
    def predict(self, 
                messages: List[ChatAgentMessage],
                context: Optional[ChatContext] = None,
                custom_inputs: Optional[dict[str, Any]] = None,
                ) -> ChatAgentResponse:  

        response_messages = [
            chunk.delta
            for chunk in self.predict_stream(messages, context, custom_inputs)
        ]

        return ChatAgentResponse(messages=response_messages)
    

    #@mlflow.trace(name="Agent", span_type='AGENT')
    def predict_stream(self, 
                messages: List[ChatAgentMessage],
                context: Optional[ChatContext] = None,
                custom_inputs: Optional[dict[str, Any]] = None,
                ) -> Generator[ChatAgentChunk, None, None]: 

        for message in self.agent_tool_calling(messages=messages):   # Streaming as messages to ensure quality & clean formatting
            yield ChatAgentChunk(delta=message)


#########################################################
#
####  Deactivating agent creation
#
# Using the agent as a standard class to avoid .self conflicts here.
# Each user gets their own isolated Python process in Databricks Apps.
# However, when packaged as an agent during the deployment phase,
# .self values spill over (since the agent is not yet deployed).
#########################################################

#agent = ChatBotAgent()
#mlflow.models.set_model(agent)