from .llm_client import call_llm
from src.utils import obtenir_heure_formatee

def choose_tool(user_query: str, server_url: str) -> str:
    system_message = (
        "You are a routing agent. Your ONLY job is to return a tool name based on the user's request.\n"
        "TOOLS:\n"
        "- get_time: User asks for the time or date.\n"
        "- change_language_en: User asks to speak in English.\n"
        "- change_language_fr: User asks to speak in French.\n"
        "- None: Conversational request.\n\n"
        "Return ONLY the tool name."
        "example: what time is it? -> get_time"
        "example: Quelle heure est t'il ? -> get_time"
        "example: can you speak in english? -> change_language_en"
        "example: parle en français -> change_language_fr"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ]
    
    try:
        response = call_llm(
            server_url=server_url,
            messages=messages,
            max_tokens=8,
            temperature=0.1,
            top_p=1.0,
            presence_penalty=1.0
        )
        if response:
            content = response.choices[0].message.content.strip()
            print(f"🤖 [Agent Fonction] Réponse brute LLM : {content}")
            
            valid_tools = ["get_time", "change_language_en", "change_language_fr", "None"]
            for tool in valid_tools:
                if tool.lower() in content.lower():
                    return tool
            
            return "None"
        return "None"
    except Exception as e:
        return "None"

def execute_tool(tool_name: str) -> dict:
    """Exécute l'outil sélectionné de manière déterministe et retourne la réponse."""
    if tool_name == "get_time":
        heure_texte = obtenir_heure_formatee()
        return {
            "text": heure_texte,
            "action": "None",
            "display": "None"
        }
    elif tool_name == "change_language_en":
        return {
            "text": "I will now speak in English.",
            "action": "None",
            "display": "None",
            "switch_lang": "en"
        }
    elif tool_name == "change_language_fr":
        return {
            "text": "Je vais maintenant parler en français.",
            "action": "None",
            "display": "None",
            "switch_lang": "fr"
        }
    return None
