from .llm_client import call_llm

def get_chat_response(history: list, current_user_text: str, context_info: str, server_url: str) -> str:
    system_content = f"Tu es un robot de compagnie. Parle français, répond brièvement. \n aide toi du Contexte pour répondre: {context_info}"

    messages = [{"role": "system", "content": system_content}]

    recent_history = history[-3:] 
    for msg in recent_history:
        role = "assistant" if msg['role'] == "assistant" else "user"
        messages.append({"role": role, "content": msg['content']})

    messages.append({"role": "user", "content": current_user_text})

    t, tp, pp = 0.4, 1.0, 2.0
        
    try:
        response = call_llm(
            server_url=server_url,
            messages=messages,
            max_tokens=1024,
            temperature=t,
            top_p=tp,
            presence_penalty=pp,
            stream=True
        )
        
        if not response:
            return "Erreur Timeout."

        print("\n🤖 [LLM Génération] : ", end="", flush=True)
        full_response = ""
        for chunk in response:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                
                # On affiche les pensées si le modèle insiste pour réfléchir
                if getattr(delta, "reasoning_content", None):
                    print(f"\033[90m{delta.reasoning_content}\033[0m", end="", flush=True)
                    
                # On affiche la réponse finale
                if getattr(delta, "content", None):
                    print(delta.content, end="", flush=True)
                    full_response += delta.content
        
        print() # Retour à la ligne final
        return full_response.strip()

    except Exception as e:
        print(f"❌ Erreur Chat: {e}")
        return "Désolé, j'ai eu un problème de connexion."
