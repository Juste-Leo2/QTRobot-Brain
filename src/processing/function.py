# src/processing/function.py
import requests
import json

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
    :param user_query: La question de l'utilisateur.
    :param server_url: L'URL du serveur de complétion.
    :param headers: Les en-têtes de la requête HTTP.
    :return: La fonction choisie sous forme de chaîne de caractères.
    """
    prompt = build_tool_prompt(user_query, AVAILABLE_TOOLS)
    
    payload = {
        "prompt": prompt,
        "temperature": 0.0,
        "n_predict": 32,
        "stop": ["\n"]
    }

    response = requests.post(server_url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()

    response_data = response.json()
    return response_data['content'].strip()

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
            break
        except KeyboardInterrupt:
            print("\nAu revoir !")
            break

if __name__ == "__main__":
    main_function_loop()