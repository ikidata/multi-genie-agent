import json
import os
import requests 
from general_functions import run_rest_api

def post_genie(server_hostname: str, token: str, genie_space_id: str, prompt: str) -> dict:
    try:
        api_command = f'/genie/spaces/{genie_space_id}/start-conversation'
        payload = {'content': prompt}
        response = run_rest_api(server_hostname = server_hostname, token = token, api_version = '2.0', api_command = api_command, action_type = 'POST', payload = payload) 
    except Exception as e:
        response = e
    return response


def get_genie_query_results(server_hostname: str, token: str, genie_space_id: str, conversation_id: str, message_id: str) -> dict:
    try:
        api_command = f'/genie/spaces/{genie_space_id}/conversations/{conversation_id}/messages/{message_id}'
        response = run_rest_api(server_hostname = server_hostname, token = token, api_version = '2.0', api_command = api_command, action_type = 'GET') 
    except Exception as e:
        response = e
    return response


def get_genie_query_attachment_results(server_hostname: str, token: str, genie_space_id: str, conversation_id: str, message_id: str, attachment_id: str) -> str:
    try:
        api_command = f'/genie/spaces/{genie_space_id}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result'
        response = run_rest_api(server_hostname = server_hostname, token = token, api_version = '2.0', api_command = api_command, action_type = 'GET') 
    except Exception as e:
        response = e
    response = response.json()['statement_response']['result']['data_array']  
    return str(response) 