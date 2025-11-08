# src/processing/chat.py
import requests
import json

# Constantes pour la construction du prompt
SYSTEM_MESSAGE = "You are a helpful assistant trained by Liquid AI."
BOS_TOKEN = "<|startoftext|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"

def build_prompt(history: list, window_size: int = 6) -> str:
    """
    Construit la chaîne de caractères finale pour le prompt à partir de l'historique.
    """
    prompt = f"{BOS_TOKEN}{IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{IM_END_TOKEN}\n"
    recent_history = history[-window_size:]
    for message in recent_history:
        role = message["role"]
        content = message["content"]
        prompt += f"{IM_START_TOKEN}{role}\n{content}{IM_END_TOKEN}\n"
    prompt += f"{IM_START_TOKEN}assistant\n"
    return prompt

def get_llm_response(history: list, server_url: str, headers: dict) -> str:
    """
    Envoie une requête au serveur LLM et retourne la réponse.
    :param history: L'historique de la conversation.
    :param server_url: L'URL du serveur de complétion.
    :param headers: Les en-têtes de la requête HTTP.
    :return: La réponse textuelle de l'assistant.
    """
    prompt_string = build_prompt(history)
    
    payload = {
        "prompt": prompt_string,
        "temperature": 0.7,
        "n_predict": 2048,
        "stop": [IM_END_TOKEN, f"{IM_START_TOKEN}user"]
    }

    response = requests.post(server_url, headers=headers, data=json.dumps(payload))
    response.raise_for_status() # Lèvera une exception en cas d'erreur HTTP

    response_data = response.json()
    return response_data['content'].strip()

def main_chat_loop():
    """Fonction principale pour une conversation interactive en console."""
    SERVER_URL = "http://localhost:8084/completion"
    HEADERS = {"Content-Type": "application/json"}
    history = []

    print("Bienvenue dans le chat avec votre LLM local.")
    print("Tapez 'exit' ou 'quit' pour quitter.")
    print("-" * 30)

    while True:
        try:
            user_input = input("Vous: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Au revoir !")
                break
            
            history.append({"role": "user", "content": user_input})
            
            assistant_response = get_llm_response(history, SERVER_URL, HEADERS)
            print(f"Assistant: {assistant_response}")
            
            history.append({"role": "assistant", "content": assistant_response})

        except requests.exceptions.RequestException as e:
            print(f"\nErreur de connexion au serveur : {e}")
            break
        except KeyboardInterrupt:
            print("\nAu revoir !")
            break

if __name__ == "__main__":
    main_chat_loop()