# src/processing/agents.py

import requests
import json
import time

# --- CONSTANTES PROMPT ---
BOS_TOKEN = "<|startoftext|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"

def call_llm(prompt, server_url, headers):
    """Helper simple pour appel LLM"""
    payload = {
        "prompt": prompt,
        "temperature": 0.1, 
        "n_predict": 64, 
        "stop": [IM_END_TOKEN, "\n"]
    }
    try:
        response = requests.post(server_url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['content'].strip()
    except:
        return None
    return None

# =================================================================
# AGENT 3 : GESTUELLE & EMOTIONS
# =================================================================

def run_agent_3_gesture(user_text, bot_response, config):
    """
    Agent 3 : Décide du mouvement du robot en fonction de la conversation.
    """
    server_url = config['llm_server']['url']
    headers = config['llm_server']['headers']
    
    gestures = ", ".join(config['qt_robot']['gestures'])
    emotions = ", ".join(config['qt_robot']['emotions'])

    system_prompt = (
        "You are the Animation Controller for the QT Robot.\n"
        "Your task: Choose the most appropriate gesture or emotion code based on the interaction.\n\n"
        f"AVAILABLE GESTURES: [{gestures}]\n"
        f"AVAILABLE EMOTIONS: [{emotions}]\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the User Input and the Robot Response.\n"
        "2. If the robot is greeting, use 'QT/hi', 'QT/wave' or 'QT/happy'.\n"
        "3. If the robot is thinking or hesitating, use 'QT/thinking'.\n"
        "4. If no specific movement is needed, return 'None'.\n"
        "5. Output ONLY the code (e.g., 'QT/happy')."
    )

    context = f"User Input: {user_text}\nRobot Response: {bot_response}"
    
    prompt = f"{BOS_TOKEN}{IM_START_TOKEN}system\n{system_prompt}{IM_END_TOKEN}\n"
    prompt += f"{IM_START_TOKEN}user\n{context}{IM_END_TOKEN}\n"
    prompt += f"{IM_START_TOKEN}assistant\nCode:"

    result = call_llm(prompt, server_url, headers)
    
    if not result or "None" in result:
        return None
    
    result = result.replace("Code:", "").strip()
    
    # Validation
    if result in config['qt_robot']['gestures']:
        return {'type': 'gesture', 'name': result}
    elif result in config['qt_robot']['emotions']:
        return {'type': 'emotion', 'name': result}
    
    return None

# =================================================================
# AGENT 4 : AFFICHAGE (DISPLAY) - CORRIGÉ
# =================================================================

def run_agent_4_display(user_text, bot_response, config):
    """
    Agent 4 : Décide ce qui doit être affiché sur la tablette.
    Prompt renforcé avec des exemples (Few-Shot) pour éviter les hallucinations.
    """
    server_url = config['llm_server']['url']
    headers = config['llm_server']['headers']

    system_prompt = (
        "You are the Display Controller for the QT Robot tablet.\n"
        "Your task: Decide what to show on the screen based on the interaction.\n\n"
        "INSTRUCTIONS:\n"
        "1. FORMAT: You must start with 'IMAGE:' or 'TEXT:'.\n"
        "2. DEFAULT: If the interaction is social (hello, chat), use 'IMAGE: QT/happy'.\n"
        "3. TEXT: Use 'TEXT:' only if the user EXPLICITLY asks to read/see something or if showing data (time, numbers).\n"
        "4. IMAGE: Use 'IMAGE:' to show faces (QT/happy, QT/sad) or objects requested.\n\n"
        "EXAMPLES:\n"
        "Context: User 'Hello', Robot 'Hi there!'\n"
        "Output: IMAGE: QT/happy\n\n"
        "Context: User 'Show me a cat', Robot 'Here is a cat.'\n"
        "Output: IMAGE: cat.jpg\n\n"
        "Context: User 'Write Welcome', Robot 'Ok.'\n"
        "Output: TEXT: Welcome\n\n"
        "Context: User 'What time is it?', Robot 'It is 8pm.'\n"
        "Output: TEXT: 20:00\n\n"
        "Context: User 'I am sad', Robot 'Oh no.'\n"
        "Output: IMAGE: QT/sad"
    )

    context = f"User: '{user_text}', Robot: '{bot_response}'"

    prompt = f"{BOS_TOKEN}{IM_START_TOKEN}system\n{system_prompt}{IM_END_TOKEN}\n"
    prompt += f"{IM_START_TOKEN}user\n{context}{IM_END_TOKEN}\n"
    prompt += f"{IM_START_TOKEN}assistant\nOutput:" # On force le début de la réponse

    result = call_llm(prompt, server_url, headers)

    if not result or "None" in result:
        # Fallback par défaut si l'agent hésite
        return {'type': 'image', 'content': 'QT/happy'}

    # Nettoyage si le LLM répète "Output:"
    result = result.replace("Output:", "").strip()

    # Parsing plus robuste
    if "TEXT:" in result:
        # On coupe au premier 'TEXT:' pour récupérer la suite
        content = result.split("TEXT:", 1)[1].strip()
        return {'type': 'text', 'content': content}
    
    elif "IMAGE:" in result:
        content = result.split("IMAGE:", 1)[1].strip()
        return {'type': 'image', 'content': content}
    
    # Si le LLM renvoie juste "QT/happy" sans préfixe (ça arrive)
    if "QT/" in result:
        return {'type': 'image', 'content': result}

    # Par défaut
    return {'type': 'image', 'content': 'QT/happy'}