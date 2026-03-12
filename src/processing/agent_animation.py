from .llm_client import call_llm
import json

def extract_json_from_response(text):
    text = text.strip()
    start_idx = text.find("{")
    end_idx = text.rfind("}")

    if start_idx != -1 and end_idx != -1:
        clean_json = text[start_idx : end_idx + 1]
    else:
        return {"action": "None", "display": "None"}

    try:
        data = json.loads(clean_json)
        act = data.get("action", "None")
        disp = data.get("display", "None")
        
        if act and act != "None" and not act.startswith("QT/"): act = f"QT/{act}"
        if disp and disp != "None" and not disp.startswith("QT/"): disp = f"QT/{disp}"

        return {"action": act, "display": disp}
    except json.JSONDecodeError:
        return {"action": "None", "display": "None"}

def get_animation(text_response: str, server_url: str) -> dict:
    system_message = """You are an animation extractor.
Extract the most appropriate action and display based on the assistant's response.
ACTIONS: hi, bye, wave, nod, shake_head, thinking, surprise, angry, kiss, None
DISPLAYS: happy, sad, angry, surprise, neutral, None

Respond ONLY with a JSON using this exact format:
{
  "action": "value",
  "display": "value"
}"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": text_response}
    ]
    
    try:
        response = call_llm(
            server_url=server_url,
            messages=messages,
            max_tokens=100,
            temperature=0.1,
            top_p=1.0,
            presence_penalty=2.0
        )
        if response:
            content = response.choices[0].message.content.strip()
            return extract_json_from_response(content)
        return {"action": "None", "display": "None"}
    except Exception as e:
        print(f"❌ Erreur Animation: {e}")
        return {"action": "None", "display": "None"}
