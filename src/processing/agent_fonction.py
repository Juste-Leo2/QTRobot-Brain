from .llm_client import call_llm

def choose_tool(user_query: str, server_url: str) -> str:
    system_message = (
        "You are a routing agent. Your ONLY job is to return a tool name based on the user's request.\n"
        "TOOLS:\n"
        "- get_time: User asks for the time or date.\n"
        "- None: Conversational request.\n\n"
        "Return ONLY the tool name."
        "example: what time is it? -> get_time"
        "example: Quelle heure est t'il ? -> get_time"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ]
    
    try:
        response = call_llm(
            server_url=server_url,
            messages=messages,
            max_tokens=10,
            temperature=0.1,
            top_p=1.0,
            presence_penalty=2.0
        )
        if response:
            content = response.choices[0].message.content.strip()
            print(f"🤖 [Agent Fonction] Réponse brute LLM : {content}")
            
            valid_tools = ["get_time", "None"]
            for tool in valid_tools:
                if tool.lower() in content.lower():
                    return tool
            
            return "None"
        return "None"
    except Exception as e:
        return "None"
