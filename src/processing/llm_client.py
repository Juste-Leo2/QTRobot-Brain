import time
import json
import requests
from types import SimpleNamespace

def call_llm(
    server_url: str,
    messages: list,
    max_tokens: int,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    extra_body: dict = None,
    stream: bool = False
):
    if extra_body is None:
        extra_body = {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0
        }
    
    # Convertir les messages au format ChatML
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        content_str = str(content)
        prompt += f"<|im_start|>{role}\n{content_str}<|im_end|>\n"
    
    # Injection du tag de raisonnement pour l'assistant
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n"

    url = f"{server_url}/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer sk-no-key-required"}
    
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "stream": stream,
        **extra_body
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, stream=stream)
            response.raise_for_status()

            if stream:
                def generate():
                    for line in response.iter_lines():
                        if line:
                            line_decode = line.decode('utf-8')
                            if line_decode.startswith("data: "):
                                data_str = line_decode[6:]
                                if data_str.strip() == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    # Mapping pour compatibilité avec /chat/completions (delta.content)
                                    if "choices" in chunk and len(chunk["choices"]) > 0:
                                        if "text" in chunk["choices"][0]:
                                            text_val = chunk["choices"][0].pop("text")
                                            chunk["choices"][0]["delta"] = {"content": text_val}
                                    
                                    yield json.loads(json.dumps(chunk), object_hook=lambda d: SimpleNamespace(**d))
                                except json.JSONDecodeError:
                                    pass
                return generate()
            else:
                resp = json.loads(response.text)
                # Mapping pour compatibilité avec /chat/completions (message.content)
                if "choices" in resp and len(resp["choices"]) > 0:
                    if "text" in resp["choices"][0]:
                        text_val = resp["choices"][0].pop("text")
                        resp["choices"][0]["message"] = {"content": text_val}
                        
                return json.loads(json.dumps(resp), object_hook=lambda d: SimpleNamespace(**d))

        except Exception as e:
            if "Connection" in str(e) or "503" in str(e):
                time.sleep(2)
                continue
            raise e
            
    return None
