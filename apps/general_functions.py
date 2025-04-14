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

def get_documentation() -> str:  
    """  
    Reads the contents of a documentation file and returns it as a string.  
  
    Returns:  
        str: The content of the documentation file.  
    """  
      
    # Define the file name (hard-coded for demo purpose)  
    file_name = 'fabricated_documentation.md'  
      
    # Read the contents of the file  
    with open(file_name, 'r') as file:  
        documentation_content = file.read()  
      
    # Return the content of the documentation file  
    return documentation_content  