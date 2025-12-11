#!/bin/bash

# =============================================================================
#      LANCEMENT DE L'APPLICATION (LINUX)
# =============================================================================

# 1. Vérification que l'environnement virtuel existe
if [ ! -d ".venv" ]; then
    echo "Erreur: Le dossier .venv n'a pas ete trouve."
    echo "Veuillez d'abord executer './setup.sh' pour effectuer l'installation."
    echo "Appuyez sur une touche pour quitter..."
    read -n 1 -s -r
    exit 1
fi

echo "Activation de l'environnement Python..."

# 2. Activation de l'environnement
# Sous Linux, on utilise 'source' et le chemin est dans bin/
source .venv/bin/activate

# Vérification si l'activation a fonctionné
if [ $? -ne 0 ]; then
    echo "Erreur critique: Impossible d'activer l'environnement."
    exit 1
fi

# 3. Exécution des scripts
# Note: Une fois activé, "python" pointe automatiquement vers celui du .venv
# C'est l'équivalent de "uv run" mais plus direct ici.

echo "Lancement du telechargement des modeles..."
python src/download_model.py --ci-mode

echo "Lancement de l'application principale..."
python main.py

# 4. Nettoyage
echo "Fermeture..."
deactivate
