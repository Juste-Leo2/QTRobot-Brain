# src/processing/chat.py

import requests
import json
import yaml
import os
import time
import re

# --- CONFIGURATION TOKENS ---
BOS_TOKEN = "<|startoftext|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"

def extract_json_from_response(text):
    """
    Extrait le JSON de manière robuste en cherchant les accolades { et }.
    Ignore le texte avant (doublons markdown) et après.
    """
    text = text.strip()
    
    # 1. Méthode "Chercheur d'Or" : On cherche juste le bloc JSON valide
    start_idx = text.find("{")
    end_idx = text.rfind("}")

    if start_idx != -1 and end_idx != -1:
        clean_json = text[start_idx : end_idx + 1]
    else:
        # Pas d'accolades trouvées, c'est probablement du texte brut
        return {"Response": text, "action": "None", "display": "None"}

    try:
        data = json.loads(clean_json)
        
        # --- RECONSTRUCTION DES PREFIXES QT/ ---
        act = data.get("action", "None")
        disp = data.get("display", "None")
        
        # On remet "QT/" si le LLM l'a oublié (pour économiser des tokens)
        if act and act != "None" and not act.startswith("QT/"):
            act = f"QT/{act}"
            
        if disp and disp != "None" and not disp.startswith("QT/"):
            disp = f"QT/{disp}"

        return {
            "Response": data.get("Response", "Je n'ai pas compris."),
            "action": act,
            "display": disp
        }
    except json.JSONDecodeError:
        print(f"⚠️ Erreur Parsing JSON. Brut: {clean_json}")
        # Fallback : on renvoie tout le texte dans Response si le JSON est cassé
        return {"Response": text, "action": "None", "display": "None"}

def build_full_prompt(history: list, current_user_text: str, context_info: str) -> str:
    """
    Prompt optimisé. On garde l'amorce ```json à la fin pour forcer le mode code.
    """
    
    system_content = f"""You are a Robot companion. Speak French.

ACTIONS: hi, bye, wave, nod, shake_head, thinking, surprise, angry, kiss, None
DISPLAYS: happy, sad, angry, surprise, neutral, None
CONTEXT: {context_info}

INSTRUCTIONS:
1. Reply in "Response" (French).
2. Pick 1 action & 1 display from lists.
3. JSON format only.

EXAMPLE:
```json
{{
  "Response": "Salut ! Content de te voir.",
  "action": "wave",
  "display": "happy"
}}
```"""

    prompt = f"{BOS_TOKEN}{IM_START_TOKEN}system\n{system_content}{IM_END_TOKEN}\n"

    recent_history = history[-3:] 
    for msg in recent_history:
        role = "assistant" if msg['role'] == "assistant" else "user"
        content = msg['content']
        prompt += f"{IM_START_TOKEN}{role}\n{content}{IM_END_TOKEN}\n"

    prompt += f"{IM_START_TOKEN}user\n{current_user_text}{IM_END_TOKEN}\n"
    
    # On force le début du bloc code. 
    # Le parseur s'en fiche maintenant si le LLM le répète ou non.
    prompt += f"{IM_START_TOKEN}assistant\n```json"
    
    return prompt

def get_multimodal_response(history: list, current_user_text: str, context_info: str, server_url: str, headers: dict) -> dict:
    prompt_string = build_full_prompt(history, current_user_text, context_info)
    
    payload = {
        "prompt": prompt_string,
        "temperature": 0.3,
        "n_predict": 1024,
        "min_p": 0.15,
        "repetition_penalty": 1.05,
        "stop": [IM_END_TOKEN, "```\n"] 
    }

    max_retries = 30
    for attempt in range(max_retries):
        try:
            response = requests.post(server_url, headers=headers, json=payload)
            
            if response.status_code == 503:
                print(f"⏳ Chargement modèle... ({attempt+1}/{max_retries})")
                time.sleep(2)
                continue
            
            response.raise_for_status()
            response_data = response.json()
            
            # --- CORRECTION ---
            # On ne rajoute PLUS manuellement "```json" ici.
            # On passe le contenu brut. Extract_json trouvera les accolades.
            return extract_json_from_response(response_data['content'])

        except requests.exceptions.ConnectionError:
            print(f"⏳ Connexion serveur... ({attempt+1}/{max_retries})")
            time.sleep(2)
            continue
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return {"Response": f"Erreur: {e}", "action": "None", "display": "QT/sad"}

    return {"Response": "Erreur Timeout.", "action": "None", "display": "QT/sad"}