# src/processing/api_google.py

from google import genai
import json
import re

class GoogleGeminiHandler:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        # UTILISATION STRICTE DE GEMMA 3 27B
        self.model_name = "gemma-3-27b-it" 

    def _clean_json(self, text):
        """
        Nettoie la réponse pour extraire le JSON proprement, 
        même si le modèle ajoute du texte autour.
        """
        text = text.strip()
        
        # 1. Recherche d'un bloc markdown ```json ... ```
        if "```json" in text:
            # On prend tout ce qui est entre ```json et ```
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
             # Cas où il oublie le mot 'json' mais met les backticks
            text = text.split("```")[1]
            
        return text.strip()

    def router_api(self, user_text):
        """
        Agent 1 (API) : Sélectionne l'outil.
        """
        prompt = (
            "You are a router agent. Analyze the user request and return ONLY the tool name.\n"
            "TOOLS:\n"
            "- get_vision: User wants to see/describe/analyze an image.\n"
            "- get_time: User asks for time/date.\n"
            "- None: Normal conversation.\n\n"
            f"User Request: {user_text}\n"
            "Tool Name:"
        )

        try:
            # Pas de JSON mode ici, juste du texte brut
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            result = response.text.strip()
            
            # Nettoyage au cas où il soit bavard
            for tool in ["get_vision", "get_time", "None"]:
                if tool in result:
                    return tool
            return "None"
        except Exception as e:
            print(f"❌ Erreur API Router: {e}")
            return "None"

    def generate_fused_response(self, user_text, history, tool_result, config):
        """
        Agent Fusionné (API) : Génère Texte + Geste + Display en un seul appel.
        """
        gestures = str(config['qt_robot']['gestures'])
        emotions = str(config['qt_robot']['emotions'])
        
        short_history = history[-3:] if len(history) > 3 else history
        # Conversion simple en string pour le prompt
        history_str = str(short_history)

        system_prompt = (
            "You are QT Robot, a helpful assistant. You control your speech, body, and screen.\n"
            "Your goal: Generate a JSON response containing your answer, body movement, and screen display.\n\n"
            "--- CONFIGURATION ---\n"
            f"AVAILABLE GESTURES: {gestures}\n"
            f"AVAILABLE EMOTIONS: {emotions}\n"
            "SCREEN RULES: Use 'QT/happy' for default, or 'TEXT:...' for info.\n\n"
            "--- INPUT CONTEXT ---\n"
            f"Conversation History: {history_str}\n"
            f"User Last Input: {user_text}\n"
            f"Tool Result (Context): {tool_result}\n\n"
            "--- OUTPUT FORMAT ---\n"
            "You MUST answer strictly using this JSON format inside a markdown block:\n"
            "```json\n"
            "{\n"
            '  "response_text": "Your answer here",\n'
            '  "robot_action": "QT/happy",\n'
            '  "screen_display": "QT/happy"\n'
            "}\n"
            "```"
        )

        try:
            # RETRAIT DU PARAMETRE 'response_mime_type' QUI FAISAIT PLANTER GEMMA 3
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=system_prompt
            )
            
            raw_text = self._clean_json(response.text)
            
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                print(f"⚠️ Erreur parsing JSON Gemma: {raw_text}")
                # Tentative de rattrapage manuel ou fallback
                return {
                    "text": raw_text[:200], # On renvoie le texte brut au pire
                    "action": "None",
                    "display": "QT/happy"
                }
            
            return {
                "text": data.get("response_text", "Je n'ai pas compris."),
                "action": data.get("robot_action", "None"),
                "display": data.get("screen_display", "QT/happy")
            }

        except Exception as e:
            print(f"❌ Erreur API Generation: {e}")
            return {
                "text": "Désolé, j'ai eu un problème de connexion API.",
                "action": "QT/sad",
                "display": "QT/sad"
            }