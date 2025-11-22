# src/processing/function.py

import requests
import json
import time
import os
import yaml

# --- CONSTANTES (Format ChatML pour structurer le prompt) ---
BOS_TOKEN = "<|startoftext|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"

def load_config():
    """Charge la configuration depuis le fichier YAML."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return None

def build_tool_prompt(user_query: str, tools: list = None) -> str:
    """
    Construit un prompt très directif avec des exemples (Few-Shot) 
    pour forcer le choix entre get_vision, get_time ou None.
    
    Note: L'argument 'tools' est gardé pour la compatibilité de signature, 
    mais les outils sont maintenant hardcodés dans le prompt comme demandé.
    """
    
    # Définition stricte des outils et des exemples pour guider le LLM
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

    # Construction du prompt avec les balises
    prompt = f"{BOS_TOKEN}{IM_START_TOKEN}system\n{system_message}{IM_END_TOKEN}\n"
    
    # La requête actuelle de l'utilisateur
    prompt += f"{IM_START_TOKEN}user\n{user_query}{IM_END_TOKEN}\n"
    
    # Force le début de la réponse pour que le LLM ne complète que le mot manquant
    prompt += f"{IM_START_TOKEN}assistant\nTool:"
    
    return prompt

def choose_tool(user_query: str, server_url: str, headers: dict) -> str:
    """
    Interroge le LLM pour choisir l'outil approprié.
    Entrées et sorties conservées pour la portabilité.
    """
    # On appelle le constructeur de prompt (tools est ignoré dedans)
    prompt = build_tool_prompt(user_query)
    
    payload = {
        "prompt": prompt,
        "temperature": 0.0,      # Zéro créativité requise pour du routage
        "n_predict": 10,         # On attend juste un mot
        "stop": [IM_END_TOKEN, "\n", "User:"] # Arrêts stricts
    }

    # --- Gestion du Retry (identique à chat.py) ---
    max_retries = 30
    
    for attempt in range(max_retries):
        try:
            # Utilisation de json=payload pour la cohérence
            response = requests.post(server_url, headers=headers, json=payload)
            
            if response.status_code == 503:
                # Modèle en chargement
                time.sleep(2)
                continue

            response.raise_for_status()

            response_data = response.json()
            
            # Nettoyage de la réponse
            content = response_data['content'].strip()
            
            # Sécurité : si le LLM répète "Tool: get_time", on nettoie
            if content.startswith("Tool:"):
                content = content.replace("Tool:", "").strip()
            
            # Sécurité : on s'assure que c'est un des mots clés attendus, sinon None
            valid_tools = ["get_vision", "get_time", "None"]
            # On peut être tolérant sur la casse ou les espaces
            for tool in valid_tools:
                if tool.lower() in content.lower():
                    return tool
            
            # Si le LLM a répondu quelque chose d'inattendu (ex: "I don't know"), on renvoie None par sécurité
            return "None"
            
        except requests.exceptions.ConnectionError:
            time.sleep(2)
            continue
        except Exception as e:
            # En cas d'erreur critique, on renvoie None ou on raise (selon préférence)
            # Ici on raise pour le debug comme demandé
            raise e

    raise requests.exceptions.HTTPError("Timeout: Serveur injoignable.")

def main_function_loop():
    """Boucle principale pour tester la sélection de fonction en console."""
    
    config = load_config()
    
    if config and 'llm_server' in config:
        SERVER_URL = config['llm_server']['url']
        HEADERS = config['llm_server']['headers']
    else:
        # Fallback
        SERVER_URL = "http://localhost:8084/completion"
        HEADERS = {"Content-Type": "application/json"}
    
    print("Assistant de sélection (get_vision / get_time / None).")
    print(f"Connecté à : {SERVER_URL}")
    print("Tapez 'exit' pour quitter.")
    print("-" * 30)

    while True:
        try:
            user_input = input("Votre question : ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            chosen_function = choose_tool(user_input, SERVER_URL, HEADERS)
            
            # Affichage visuel pour confirmer le bon choix
            if chosen_function == "get_vision":
                print(f"-> 👁️  Outil choisi : {chosen_function}")
            elif chosen_function == "get_time":
                print(f"-> ⏰ Outil choisi : {chosen_function}")
            else:
                print(f"-> ❌ Outil choisi : {chosen_function}")
            print("")

        except requests.exceptions.RequestException as e:
            print(f"Erreur connexion : {e}")
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Erreur : {e}")

if __name__ == "__main__":
    main_function_loop()