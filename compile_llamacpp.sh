#!/bin/bash

# Arrêt immédiat en cas d'erreur
set -e

# LA VERSION CIBLE (Celle que tu m'as donnée)
TARGET_VERSION="b6987"

echo "🚀 COMPILATION CIBLÉE LLAMA.CPP (Version : $TARGET_VERSION)"
echo "------------------------------------------------------------"

# 1. Vérification des outils
echo "📦 Vérification des dépendances..."
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl libcurl4-openssl-dev

# 2. Dossier de travail
TARGET_DIR="models/llama_cpp"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# 3. Gestion des sources GIT
if [ -d "llama.cpp_source" ]; then
    echo "🔄 Dossier source existant trouvé."
    cd llama.cpp_source
    
    # On nettoie les modifs locales pour éviter les conflits
    echo "   Reset des modifications locales..."
    git reset --hard
    
    # On récupère tout l'historique pour trouver la version
    echo "   Téléchargement de l'historique git..."
    git fetch --all --tags
else
    echo "📥 Clonage complet du dépôt..."
    git clone https://github.com/ggml-org/llama.cpp llama.cpp_source
    cd llama.cpp_source
fi

# 4. PASSAGE A LA VERSION SPECIFIQUE
echo "🔙 ROLLBACK vers la version : $TARGET_VERSION"
# Checkout permet de revenir dans le passé à ce commit précis
git checkout "$TARGET_VERSION"

# 5. Mise à jour des sous-modules pour CETTE version spécifique
echo "📦 Mise à jour des sous-modules..."
git submodule update --init --recursive

# 6. Nettoyage du build
echo "🧹 Préparation du dossier build..."
rm -rf build
mkdir build
cd build

# 7. Configuration CMake
# Note : Sur les vieilles versions, DLLAMA_CURL n'existait peut-être pas encore.
# On garde les flags standards optimisés pour ton i5.
echo "⚙️  Configuration CMake..."
cmake .. \
    -DGGML_NATIVE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_CURL=ON \
    -DCMAKE_BUILD_TYPE=Release

# 8. Compilation
echo "🔨 Compilation en cours (Version $TARGET_VERSION)..."
cmake --build . --config Release -j $(nproc)

# 9. Installation
echo "🚚 Copie des exécutables..."

# Gestion des noms (server vs llama-server selon l'époque)
if [ -f "bin/llama-server" ]; then
    NEW_BIN="bin/llama-server"
elif [ -f "bin/server" ]; then
    NEW_BIN="bin/server"
else
    echo "❌ ERREUR : Binaire introuvable dans bin/. Vérifie les logs de compilation."
    exit 1
fi

# Retour au dossier models/llama_cpp
cd ../..

# Copie et droits
cp "llama.cpp_source/build/$NEW_BIN" ./llama-server
chmod +x ./llama-server
echo "   ✅ llama-server restauré en version $TARGET_VERSION."

# Copie pour la vision
cp ./llama-server ./llama-server-vision
echo "   ✅ llama-server-vision restauré."

echo "------------------------------------------------------------"
echo "🎉 SUCCÈS ! Tu es maintenant sur la version $TARGET_VERSION."
echo "   Tu peux relancer : python3 main.py --QT"