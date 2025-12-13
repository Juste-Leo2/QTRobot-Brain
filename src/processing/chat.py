# src/processing/chat.py

import requests
import json
import base64
import yaml
import os
import time

# Constantes (Inchangées)
SYSTEM_MESSAGE = "Tu es une IA utile et bienveillante qui aide l'utilisateur, tu disposera d'information pour guider tes réponses. Tes réponses doivent être COURTES"
BOS_TOKEN = "<|startoftext|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_prompt(history: list, window_size: int = 6) -> str:
    """
    Construit le prompt avec une fenêtre glissante (window_size) pour garder le contexte
    sans dépasser la limite de tokens.
    """
    prompt = f"{BOS_TOKEN}{IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{IM_END_TOKEN}\n"
    # On ne garde que les 'window_size' derniers messages
    recent_history = history[-window_size:]
    
    for message in recent_history:
        role = message["role"]
        content = message["content"]
        prompt += f"{IM_START_TOKEN}{role}\n{content}{IM_END_TOKEN}\n"
    prompt += f"{IM_START_TOKEN}assistant\n"
    return prompt

def get_llm_response(history: list, server_url: str, headers: dict) -> str:
    """
    Envoie l'historique au LLM texte et récupère la réponse.
    Intègre une gestion des erreurs 503 (Modèle en cours de chargement).
    """
    prompt_string = build_prompt(history)
    
    # Logique de génération STRICTEMENT conservée
    payload = {
        "prompt": prompt_string,
        "temperature": 0.2,
        "n_predict": 2048,
        "stop": [IM_END_TOKEN, f"{IM_START_TOKEN}user"]
    }

    # --- AJOUT : Boucle de tentative (Retry) pour gérer le chargement du modèle ---
    max_retries = 30 # 30 tentatives * 2sec = 60 secondes max d'attente
    
    for attempt in range(max_retries):
        try:
            response = requests.post(server_url, headers=headers, json=payload)
            
            # Si le serveur répond 503, c'est qu'il charge le modèle
            if response.status_code == 503:
                print(f"⏳ Le modèle est en cours de chargement... (Tentative {attempt+1}/{max_retries})")
                time.sleep(2)
                continue
            
            # Pour toute autre erreur HTTP (404, 500, etc.), on lève l'exception
            response.raise_for_status()
            
            response_data = response.json()
            return response_data['content'].strip()

        except requests.exceptions.ConnectionError:
            # Si le serveur n'est pas encore lancé (connexion refusée)
            print(f"⏳ En attente de la connexion au serveur... (Tentative {attempt+1}/{max_retries})")
            time.sleep(2)
            continue
            
        except Exception as e:
            return f"❌ Erreur inattendue : {e}"

    return "❌ Erreur : Le serveur LLM ne répond pas (Timeout)."


# --- PARTIE VISION (LOGIQUE CONSERVÉE SELON VOTRE DEMANDE) ---

def get_llm_response_vision(server_url: str, image_path: str, prompt: str = "Describe this image") -> str:
    """
    Envoie une requête au serveur LLM vision.
    Logique strictement identique à main.py.
    """
    try:
        # 1. Lire et encoder l'image en base64
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        return f"❌ Erreur : Image non trouvée à {image_path}"

    # 2. Construire le payload
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
    }

    # 3. Envoyer la requête avec mécanisme de retry
    max_retries = 30
    for attempt in range(max_retries):
        try:
            # Note: Le serveur vision peut ne pas avoir besoin de headers spécifiques selon la config,
            # mais requests.post gère très bien l'absence de headers si non fournis.
            response = requests.post(server_url, json=payload)
            
            if response.status_code == 503:
                print(f"⏳ Vision : Modèle en chargement... (Tentative {attempt+1}/{max_retries})")
                time.sleep(2)
                continue
                
            response.raise_for_status()
            
            # 4. Afficher la réponse
            result = response.json()
            # Structure standard OpenAI-like pour la vision
            return result['choices'][0]['message']['content']

        except requests.exceptions.ConnectionError:
            time.sleep(2)
            continue
        except Exception as e:
            return f"❌ Erreur Vision : {e}"
            
    return "❌ Erreur : Impossible de contacter le serveur vision après plusieurs tentatives."

###############################################


def main_chat_loop():
    """Fonction principale pour une conversation interactive en console."""
    try:
        config = load_config()
        server_config = config['llm_server']
        SERVER_URL = server_config['url']
        HEADERS = server_config['headers']
    except (FileNotFoundError, KeyError) as e:
        print(f"Erreur lors du chargement de la configuration : {e}")
        return

    history = []

    print("Bienvenue dans le chat avec votre LLM local.")
    print(f"Connecté à : {SERVER_URL}")
    print("Tapez 'exit' ou 'quit' pour quitter.")
    print("-" * 30)

    while True:
        try:
            user_input = input("Vous: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Au revoir !")
                break
            
            history.append({"role": "user", "content": user_input})
            
            # L'appel gère maintenant les 503 automatiquement
            assistant_response = get_llm_response(history, SERVER_URL, HEADERS)
            print(f"Assistant: {assistant_response}")
            
            history.append({"role": "assistant", "content": assistant_response})

        except KeyboardInterrupt:
            print("\nAu revoir !")
            break
        except Exception as e:
            print(f"\nUne erreur inattendue est survenue : {e}")
            break

if __name__ == "__main__":
    main_chat_loop()