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
        Nettoie la réponse pour extraire le JSON proprement.
        """
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1]
        return text.strip()

    def router_api(self, user_text):
        """
        Agent 1 (API) : Routeur simple.
        """
        prompt = (
            "Analyze the user request and return ONLY the tool name.\n"
            "TOOLS:\n"
            "- get_vision: User wants to see/describe an image.\n"
            "- get_time: User asks for time/date.\n"
            "- None: Normal conversation.\n\n"
            f"User Request: {user_text}\n"
            "Tool Name:"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            result = response.text.strip()
            for tool in ["get_vision", "get_time", "None"]:
                if tool in result: return tool
            return "None"
        except Exception as e:
            print(f"❌ Erreur API Router: {e}")
            return "None"

    def generate_fused_response(self, user_text, history, tool_result, config):
        """
        Agent Fusionné : Génère une réponse courte + une émotion dominante.
        """
        # On récupère les listes du config pour que le LLM sache quoi choisir
        gestures = str(config['qt_robot']['gestures'])
        emotions = str(config['qt_robot']['emotions'])
        
        short_history = history[-3:] if len(history) > 3 else history
        history_str = str(short_history)

        system_prompt = (
            "You are QT Robot, a companion for children and adults. You speak French.\n"
            "**INSTRUCTIONS:**\n"
            "1. Answer in **FRENCH**, keep it **SHORT, SIMPLE and NATURAL** (1 or 2 sentences max).\n"
            "2. Choose the best **EMOTION** from the list below to match your answer.\n"
            "3. Select a **GESTURE** only if specifically needed (like saying goodbye), otherwise use an emotion for the body action too.\n\n"
            "--- AVAILABLE ACTIONS ---\n"
            f"Emotions: {emotions}\n"
            f"Gestures: {gestures}\n\n"
            "--- CONTEXT ---\n"
            f"History: {history_str}\n"
            f"User Input: {user_text}\n"
            f"Tool Context: {tool_result}\n\n"
            "--- OUTPUT FORMAT (Strict JSON) ---\n"
            "```json\n"
            "{\n"
            '  "response_text": "Ta réponse courte en français ici.",\n'
            '  "robot_action": "QT/happy",\n'  # Peut être une émotion ou un geste
            '  "screen_display": "QT/happy"\n' # Doit être une émotion ou TEXT:...
            "}\n"
            "```"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=system_prompt
            )
            
            raw_text = self._clean_json(response.text)
            
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                print(f"⚠️ Erreur parsing JSON: {raw_text}")
                return {
                    "text": raw_text[:200], 
                    "action": "QT/happy",
                    "display": "QT/happy"
                }
            
            return {
                "text": data.get("response_text", "Je n'ai pas compris."),
                "action": data.get("robot_action", "QT/neutral"),
                "display": data.get("screen_display", "QT/neutral")
            }

        except Exception as e:
            print(f"❌ Erreur API Generation: {e}")
            return {
                "text": "J'ai un petit souci technique.",
                "action": "QT/sad",
                "display": "QT/sad"
            }