import requests
import json
import time

class OpenRouterHandler:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model_name = "meta-llama/llama-3.3-70b-instruct:free"
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self._last_call_ts = 0
        self._min_delay = 1.0  # 1 seconde minimum entre chaque appel

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

    def _chat(self, prompt):
        """
        Appel REST vers l'API OpenRouter.
        Attend 1s minimum entre chaque appel pour respecter le rate limit.
        """
        # Cooldown entre les appels
        elapsed = time.time() - self._last_call_ts
        if elapsed < self._min_delay:
            time.sleep(self._min_delay - elapsed)

        response = requests.post(
            url=self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/QTRobot-Brain",
                "X-OpenRouter-Title": "QTRobot-Brain",
            },
            data=json.dumps({
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }),
            timeout=30
        )
        self._last_call_ts = time.time()
        
        if response.status_code == 429:
            print(f"⚠️ Rate limit OpenRouter — Détails: {response.text}")
        
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def router_api(self, user_text):
        """
        Agent 1 (API) : Routeur simple.
        """
        prompt = (
            "Analyze the user request and return ONLY the tool name.\n"
            "TOOLS:\n"
            "- get_time: User asks for time/date.\n"
            "- None: Normal conversation.\n\n"
            f"User Request: {user_text}\n"
            "Tool Name:"
        )

        try:
            result = self._chat(prompt).strip()
            for tool in ["get_time", "None"]:
                if tool in result: return tool
            return "None"
        except Exception as e:
            print(f"❌ Erreur OpenRouter Router: {e}")
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
            '  "robot_action": "QT/happy",\n'
            '  "screen_display": "QT/happy"\n'
            "}\n"
            "```"
        )

        try:
            raw_text = self._clean_json(self._chat(system_prompt))
            
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
            print(f"❌ Erreur OpenRouter Generation: {e}")
            return {
                "text": "J'ai un petit souci technique.",
                "action": "QT/sad",
                "display": "QT/sad"
            }
