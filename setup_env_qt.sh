#!/bin/bash

# =============================================================================
#      INITIALISATION DE L'ENVIRONNEMENT DE DEVELOPPEMENT (LINUX - FIX TIMEOUT)
# =============================================================================

echo ""
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

handle_error() {
    echo ""
    echo "=========================================================="
    echo -e "${RED}     ERREUR: L'INSTALLATION A ECHOUE.${NC}"
    echo "=========================================================="
    echo "Veuillez verifier les messages d'erreur ci-dessus."
    echo ""
    echo "Appuyez sur une touche pour quitter..."
    read -n 1 -s -r
    exit 1
}

# --- Etape 0: Installation des paquets système (apt) ---
echo "[0/5] Mise à jour et installation des librairies système..."

if command -v apt-get >/dev/null 2>&1; then
    echo -e "${YELLOW}Demande de privilèges sudo pour installer les dépendances système...${NC}"
    
    # Installation des pré-requis pour PyAudio, OpenCV, Tkinter, etc.
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        portaudio19-dev \
        python3-tk \
        libgl1 \
        libssl-dev \
        libffi-dev

    if [ $? -ne 0 ]; then
        echo -e "${RED}ECHEC: Impossible d'installer les paquets système.${NC}"
        handle_error
    fi
    echo "   [OK] Librairies système installées."
else
    echo -e "${YELLOW}ATTENTION: 'apt-get' non trouvé. Assurez-vous d'avoir installé les librairies manuellement.${NC}"
fi


# --- Etape 1: Verification des pre-requis ---
echo "[1/5] Verification des pre-requis fichiers..."
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}ECHEC: Le fichier 'requirements.txt' est introuvable.${NC}"
    handle_error
fi
echo "   [OK] Fichier requirements.txt trouve."


# --- Etape 2: Verification et installation de 'uv' ---
echo "[2/5] Verification et installation de 'uv'..."

if command -v uv >/dev/null 2>&1; then
    echo "   [OK] uv est deja installe."
else
    echo "   uv non trouve, lancement de l'installation..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}ECHEC: Impossible d'installer uv.${NC}"
        handle_error
    fi

    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    if ! command -v uv >/dev/null 2>&1; then
        echo -e "${RED}ECHEC: uv installé mais non accessible.${NC}"
        handle_error
    fi
    echo "   [OK] uv installe avec succes."
fi


# --- Etape 3: Creation de l'environnement Python ---
echo "[3/5] Creation de l'environnement Python (python 3.11)..."

uv venv -p 3.11

if [ $? -ne 0 ]; then
    echo -e "${RED}ECHEC: Impossible de creer l'environnement Python.${NC}"
    handle_error
fi
echo "   [OK] Environnement Python cree dans .venv."


# --- Etape 4: Activation et installation des dependances ---
echo "[4/5] Installation des dependances Python..."

if [ ! -f "./.venv/bin/activate" ]; then
    echo -e "${RED}ECHEC: Script d'activation introuvable (.venv/bin/activate).${NC}"
    handle_error
fi

source ./.venv/bin/activate

# --- MODIFICATION ICI ---
# On augmente le timeout à 1200 secondes (20 minutes) pour TensorFlow
echo -e "${YELLOW}Extension du timeout réseau pour le téléchargement de TensorFlow...${NC}"
export UV_HTTP_TIMEOUT=1200

uv pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo -e "${RED}ECHEC: Installation des dépendances échouée.${NC}"
    deactivate
    handle_error
fi

echo "   [OK] Dependances Python installees."


# --- Etape 5: Finalisation ---
echo "[5/5] Finalisation..."
deactivate

echo ""
echo "=========================================================="
echo -e "${GREEN}      ENVIRONNEMENT CONFIGURE AVEC SUCCES !${NC}"
echo "=========================================================="
echo "Pour lancer votre projet :"
echo "  source .venv/bin/activate"
echo "  python votre_script.py"
echo ""
echo "Appuyez sur une touche pour quitter..."
read -n 1 -s -r
exit 0
