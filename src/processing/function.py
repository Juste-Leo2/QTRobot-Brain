# src/processing/function.py

import requests
import time
import os
import yaml

# --- CONSTANTES ---
BOS_TOKEN = "<|startoftext|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"

def build_tool_prompt(user_query: str) -> str:
    """
    Agent 1 : Router.
    Décide uniquement si on utilise un outil technique (Vision/Heure) ou le Chat.
    """
    system_message = (
        "You are a precise tool selector. Your ONLY job is to return the correct tool name based on the user's request.\n"
        "AVAILABLE TOOLS:\n"
        "- get_vision: Use ONLY if the user asks to describe, see, look at, or analyze an image or photo.\n"
        "- get_time: Use ONLY if the user asks for the current time, date, or hour.\n"
        "- None: Use if the request is conversational (hello, how are you) or doesn't match the tools above.\n\n"
        "EXAMPLES:\n"
        "User: What time is it?\nTool: get_time\n"
        "User: Can you describe this picture?\nTool: get_vision\n"
        "User: Hello assistant.\nTool: None\n"
        "User: What do you see?\nTool: get_vision\n"
        "User: Give me the date.\nTool: get_time"
    )

    prompt = f"{BOS_TOKEN}{IM_START_TOKEN}system\n{system_message}{IM_END_TOKEN}\n"
    prompt += f"{IM_START_TOKEN}user\n{user_query}{IM_END_TOKEN}\n"
    prompt += f"{IM_START_TOKEN}assistant\nTool:"
    
    return prompt

def choose_tool(user_query: str, server_url: str, headers: dict) -> str:
    prompt = build_tool_prompt(user_query)
    
    payload = {
        "prompt": prompt,
        "temperature": 0.1,
        "n_predict": 10,
        "stop": [IM_END_TOKEN, "\n", "User:"]
    }

    max_retries = 30
    for attempt in range(max_retries):
        try:
            response = requests.post(server_url, headers=headers, json=payload)
            if response.status_code == 503:
                time.sleep(2)
                continue
            response.raise_for_status()

            response_data = response.json()
            content = response_data['content'].strip()
            
            if content.startswith("Tool:"):
                content = content.replace("Tool:", "").strip()
            
            # Note: gestures/display ne sont PAS ici, ils sont gérés en aval par Agents 3/4
            valid_tools = ["get_vision", "get_time", "None"]
            for tool in valid_tools:
                if tool.lower() in content.lower():
                    return tool
            
            return "None"
            
        except requests.exceptions.ConnectionError:
            time.sleep(2)
            continue
        except Exception:
            return "None"
    return "None"