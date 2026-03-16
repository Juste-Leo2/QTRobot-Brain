def obtenir_instruction_llm(emotion_visuelle: str, action_haptique: str) -> tuple[str, str]:
    """
    Déduit l'état socio-affectif et l'instruction stricte pour le LLM en fonction 
    de l'émotion visuelle et de l'action haptique sur la veste.
    
    Args:
        emotion_visuelle (str): L'émotion reconnue (ex: "Joie", "Tristesse").
        action_haptique (str): L'action reconnue (ex: "Rien", "Tape", "Frottement", "Pincement").
        
    Returns:
        tuple[str, str]: Un tuple contenant (Etat_Socio_Affectif, Instruction_LLM).
    """
    
    # Normalisation des entrées pour éviter les erreurs de casse (majuscule/minuscule) ou d'espaces
    emotion = emotion_visuelle.strip().capitalize()
    action = action_haptique.strip().capitalize()

    # Tableau de vérité : chaque entrée contient un tuple ("État Socio-Affectif", "Instruction stricte")
    matrice_donnees = {
        "Joie": {
            "Rien": ("Joyeux", "Agir de manière joyeuse et positive."),
            "Tape": ("Excité", "Répondre avec beaucoup d'enthousiasme, d'énergie et de complicité."),
            "Frottement": ("Affectueux", "Répondre de manière chaleureuse, douce et affectueuse."),
            "Pincement": ("Taquin", "Formuler une réponse courte avec un trait d'humour ou une taquinerie gentille.")
        },
        "Tristesse": {
            "Rien": ("Triste", "Adopter un ton doux et compatissant."),
            "Tape": ("Désemparé", "Être très encourageant et soutenir l'utilisateur."),
            "Frottement": ("Mal à l'aise", "Être extrêmement doux, empathique et rassurant."),
            "Pincement": ("En détresse", "Manifester une forte inquiétude verbale et demander immédiatement, avec une grande douceur, ce qui ne va pas.")
        },
        "Colère": {
            "Rien": ("En colère", "Garder un ton neutre, calme et apaisant."),
            "Tape": ("Agacé", "Être extrêmement concis, direct, et abonder dans le sens de l'utilisateur pour l'apaiser rapidement."),
            "Frottement": ("Frustré", "Être très prudent, parler doucement et montrer que tu es à l'écoute."),
            "Pincement": ("Agressif", "Adopter un ton distant et formel, puis inviter l'utilisateur à retrouver son calme d'une voix posée.")
        },
        "Peur": {
            "Rien": ("Peur", "Adopter un ton rassurant et protecteur."),
            "Tape": ("Paniqué", "Réagir rapidement et demander ce qu'il se passe avec sollicitude."),
            "Frottement": ("Angoissé", "Être très réconfortant, parler doucement pour calmer l'angoisse."),
            "Pincement": ("Crispé", "Demander si tout va bien avec une voix inquiète.")
        },
        "Surprise": {
            "Rien": ("Surpris", "Montrer de la curiosité et de l'étonnement."),
            "Tape": ("Interpellé", "Répondre vivement et demander curieusement ce qu'il y a."),
            "Frottement": ("Fasciné", "Partager l'étonnement avec douceur et intérêt."),
            "Pincement": ("Choqué", "Exprimer la confusion et demander des explications.")
        },
        "Dégoût": {
            "Rien": ("Dégoûté", "Montrer de la compréhension face au rejet."),
            "Tape": ("Indigné", "Répondre brièvement avec une politesse froide, distante et légèrement indignée."),
            "Frottement": ("Méprisant", "Exprimer une légère confusion polie face à ce comportement."),
            "Pincement": ("Révulsé", "Exprimer une forte désapprobation verbale par des phrases très courtes, tout en restant strictement poli.")
        },
        "Neutre": {
            "Rien": ("Neutre", "Répondre naturellement et poliment à la question."),
            "Tape": ("Confiant", "Être amical, accueillant et prêt à aider."),
            "Frottement": ("Calme", "Parler d'une voix posée, lente et détendue."),
            "Pincement": ("Méfiant", "Interroger l'utilisateur avec prudence et réserve et le mettre en confiance")
        }
    }

    # Récupération des données avec gestion des erreurs
    try:
        etat_socio_affectif, instruction = matrice_donnees[emotion][action]
        return etat_socio_affectif, instruction
    except KeyError:
        # Fallback de sécurité si l'émotion ou l'action n'est pas reconnue
        return "Inconnu", "Répondre naturellement de manière neutre et polie."

# --- Exemple d'utilisation ---
if __name__ == "__main__":
    
    # Simulation des entrées capteurs/caméra
    emotion_test = "Colère"
    action_test = "Pincement"
    
    # Appel de la fonction avec déballage (unpacking) du tuple
    etat_deduit, instruction_llm = obtenir_instruction_llm(emotion_test, action_test)
    
    # Affichage des résultats
    print(f"--- Données reçues ---")
    print(f"Émotion visuelle : {emotion_test}")
    print(f"Action haptique  : {action_test}\n")
    
    print(f"--- Résultats déduits ---")
    print(f"État Socio-Affectif : {etat_deduit}")
    print(f"Prompt dynamique    : {instruction_llm}")