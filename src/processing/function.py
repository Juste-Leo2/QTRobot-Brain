# src/processing/function.py
import requests
import json
import time  # Ajout nécessaire pour le sleep

AVAILABLE_TOOLS = [
    "get_time() # Utile pour connaître l'heure, la date ou le jour actuel.",
    "get_weather(city: str) # Utile pour obtenir la météo d'une ville spécifique.",
    "None # Utiliser si la question ne correspond à aucune fonction (salutation, conversation, etc.)."
]

def build_tool_prompt(user_query: str, tools: list) -> str:
    """
    Construit un prompt pour forcer l'IA à choisir une fonction.
    """
    system_prompt = (
        "Tu es un expert en routage de fonctions. Ta seule tâche est de choisir la fonction la plus "
        "appropriée dans une liste pour répondre à la question de l'utilisateur.\n"
        "Si aucune fonction ne correspond, tu dois répondre `None`.\n"
        "Tu dois répondre UNIQUEMENT avec le nom de la fonction, comme `get_time()` ou `None`. "
        "N'ajoute AUCUN autre texte.\n\n"
        "Fonctions disponibles :\n"
    )
    for tool in tools:
        system_prompt += f"- {tool}\n"
    
    examples = (
        "--- EXEMPLES ---\n"
        "Question: Quelle heure est-il ?\n"
        "Fonction: get_time()\n\n"
        "Question: Quel temps fait-il à Tokyo ?\n"
        "Fonction: get_weather(city: Tokyo)\n\n"
        "Question: Bonjour, comment ça va ?\n"
        "Fonction: None\n"
        "--- FIN DES EXEMPLES ---"
    )

    full_prompt = f"{system_prompt}\n{examples}\n\n--- TÂCHE ---\nQuestion: {user_query}\nFonction:"
    return full_prompt

def choose_tool(user_query: str, server_url: str, headers: dict) -> str:
    """
    Interroge le LLM pour choisir l'outil approprié pour une requête donnée.
    Intègre une gestion des erreurs 503 (Modèle en cours de chargement).
    """
    prompt = build_tool_prompt(user_query, AVAILABLE_TOOLS)
    
    payload = {
        "prompt": prompt,
        "temperature": 0.0,
        "n_predict": 32,
        "stop": ["\n"]
    }

    # --- AJOUT : Boucle de tentative (Retry) pour gérer le chargement du modèle ---
    max_retries = 30 # 60 secondes max d'attente
    
    for attempt in range(max_retries):
        try:
            # On conserve data=json.dumps(payload) comme dans votre code original
            response = requests.post(server_url, headers=headers, data=json.dumps(payload))
            
            # Si le serveur répond 503, c'est qu'il charge le modèle
            if response.status_code == 503:
                # Optionnel : print pour le debug, peut être commenté si trop verbeux
                # print(f"⏳ Fonction : Modèle en chargement... (Tentative {attempt+1}/{max_retries})")
                time.sleep(2)
                continue

            response.raise_for_status()

            response_data = response.json()
            return response_data['content'].strip()
            
        except requests.exceptions.ConnectionError:
            # Si le serveur n'est pas encore accessible du tout
            time.sleep(2)
            continue
        except Exception as e:
            # En cas d'erreur critique, on remonte l'exception ou on renvoie None
            # Pour ne pas bloquer les tests on raise l'erreur originale
            raise e

    # Si on sort de la boucle sans succès
    raise requests.exceptions.HTTPError("Timeout: Le serveur LLM est resté en 503 trop longtemps.")

def main_function_loop():
    """Boucle principale pour tester la sélection de fonction en console."""
    SERVER_URL = "http://localhost:8084/completion"
    HEADERS = {"Content-Type": "application/json"}
    
    print("Assistant de sélection de fonctions.")
    print("Tapez 'exit' ou 'quit' pour quitter.")
    print("-" * 30)

    while True:
        try:
            user_input = input("Votre question : ")
            if user_input.lower() in ["exit", "quit"]:
                print("Au revoir !")
                break
            
            chosen_function = choose_tool(user_input, SERVER_URL, HEADERS)
            print(f"-> Fonction choisie : {chosen_function}\n")

        except requests.exceptions.RequestException as e:
            print(f"\nErreur de connexion au serveur : {e}")
            # On break ici car si le serveur est down manuellement, inutile de spammer
            break
        except KeyboardInterrupt:
            print("\nAu revoir !")
            break

if __name__ == "__main__":
    main_function_loop()