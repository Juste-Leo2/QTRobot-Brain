#!/usr/bin/env python3
"""
download_model.py - Script de vérification et téléchargement des modèles IA
pour le projet robot (VERSION WINDOWS).
Vérifie la présence des modèles et les télécharge automatiquement si nécessaire.
"""

import os
import sys
import yaml
import subprocess
import tempfile
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


# Configuration
CONFIG_PATH = "config/config.yaml"
PROJECT_ROOT = Path(__file__).parent.parent

def load_config() -> Dict:
    """Charge la configuration depuis config.yaml"""
    config_file = PROJECT_ROOT / CONFIG_PATH
    if not config_file.exists():
        print(f"❌ Erreur: Fichier de configuration non trouvé: {config_file}")
        sys.exit(1)
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_curl() -> bool:
    """Vérifie si curl est disponible sur Windows (curl.exe dans le PATH)"""
    try:
        # Sur Windows, vérifie 'curl.exe'
        subprocess.run(['curl.exe', '--version'], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Fallback pour les environnements Git Bash / WSL
            subprocess.run(['curl', '--version'], 
                          capture_output=True, check=True)
            return True
        except:
            return False

def path_exists(path: str) -> bool:
    """Vérifie si un fichier ou répertoire existe sur Windows"""
    full_path = PROJECT_ROOT / path
    return full_path.exists()

def download_file(url: str, destination: Path, desc: str = "") -> bool:
    """
    Télécharge un fichier avec curl sur Windows
    
    Args:
        url: URL du fichier à télécharger
        destination: Chemin de destination (Windows)
        desc: Description pour l'affichage
    """
    print(f"📥 Téléchargement {desc}...")
    print(f"   Depuis: {url}")
    print(f"   Vers: {destination}")
    
    # Crée le répertoire parent si nécessaire
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Commande curl pour Windows
    curl_cmd = 'curl.exe' if shutil.which('curl.exe') else 'curl'
    cmd = [
        curl_cmd, '-L', '-#', '-o', str(destination),
        '--connect-timeout', '30',
        '--retry', '3',
        '--retry-delay', '5',
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            size_mb = destination.stat().st_size / 1024 / 1024
            print(f"✅ Téléchargement terminé: {desc} ({size_mb:.1f} Mo)")
            return True
        else:
            print(f"❌ Erreur lors du téléchargement {desc}")
            print(f"   stderr: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception lors du téléchargement {desc}: {e}")
        return False

def extract_zip(zip_path: Path, extract_to: Path, expected_item: str, is_directory: bool) -> bool:
    """
    Extrait une archive zip sur Windows et vérifie la présence de l'élément
    
    Args:
        zip_path: Chemin vers le fichier zip
        extract_to: Répertoire d'extraction
        expected_item: Nom de l'élément attendu (fichier ou répertoire)
        is_directory: True si l'élément attendu est un répertoire
    """
    print(f"📦 Extraction de {zip_path.name}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Vérifie si c'est le zip de llama.cpp (contient un dossier 'bin')
            has_bin_folder = any(f.startswith('bin/') or f.startswith('bin\\') for f in zip_ref.namelist())
            
            if has_bin_folder:
                # Extrait depuis le sous-dossier 'bin/'
                bin_files = [f for f in zip_ref.namelist() if f.startswith('bin/') or f.startswith('bin\\')]
                for file_info in bin_files:
                    # Enleve le préfixe 'bin/' pour l'extraction
                    file_name = file_info[4:]  # Supprime 'bin/'
                    if file_name:  # Ignore le dossier racine 'bin/'
                        source = zip_ref.open(file_info)
                        target_path = extract_to / file_name
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with target_path.open('wb') as target:
                            shutil.copyfileobj(source, target)
                extracted_item = extract_to / expected_item
            else:
                # Extraction standard
                zip_ref.extractall(extract_to)
                extracted_item = extract_to / expected_item
            
            # Vérifie que l'élément existe
            if is_directory:
                if extracted_item.exists() and extracted_item.is_dir():
                    print(f"   ✅ Répertoire extrait: {extracted_item}")
                    return True
                else:
                    print(f"   ❌ Répertoire non trouvé: {extracted_item}")
                    dirs = [d for d in extract_to.iterdir() if d.is_dir()]
                    print(f"   Répertoires trouvés: {[d.name for d in dirs]}")
                    return False
            else:
                if extracted_item.exists() and extracted_item.is_file():
                    print(f"   ✅ Fichier extrait: {extracted_item}")
                    return True
                else:
                    # Cherche récursivement
                    found_files = list(extract_to.rglob(expected_item))
                    if found_files:
                        found_path = found_files[0]
                        if not extracted_item.parent.exists():
                            extracted_item.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(found_path), str(extracted_item))
                        print(f"   📂 Fichier déplacé vers: {extracted_item}")
                        return True
                    else:
                        print(f"   ❌ Fichier {expected_item} non trouvé dans l'archive")
                        files = list(extract_to.rglob('*'))
                        print(f"   Fichiers extraits: {[f.name for f in files[:10]]}")
                        return False
            
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--ci-mode', action='store_true', 
                       help='Utilise le modèle LLM léger pour CI')
    args = parser.parse_args()

    print("=" * 70)
    print("VÉRIFICATION ET TÉLÉCHARGEMENT DES MODÈLES IA [WINDOWS]")
    print("=" * 70)
    
    # Vérifie curl
    if not check_curl():
        print("❌ Erreur: curl.exe n'est pas installé ou non disponible dans le PATH")
        print("   Installez curl depuis: https://curl.se/windows/")
        print("   ou ajoutez-le à votre PATH système.")
        sys.exit(1)
    
    # Charge la configuration
    print("Chargement de la configuration...")
    config = load_config()
    print("✅ Configuration chargée avec succès\n")
    
    # URLs des modèles (avec modèle CI plus petit)
    llm_url = "https://huggingface.co/unsloth/LFM2-8B-A1B-GGUF/resolve/main/LFM2-8B-A1B-UD-Q3_K_XL.gguf?download=true"
    
    if args.ci_mode:
        print("🚀 Mode CI activé - Utilisation du modèle LLM léger (450M)")
        llm_url = "https://huggingface.co/LiquidAI/LFM2-VL-450M-GGUF/resolve/main/LFM2-VL-450M-Q4_0.gguf?download=true"
    
    llm_url_vision = "https://huggingface.co/LiquidAI/LFM2-VL-450M-GGUF/resolve/main/LFM2-VL-450M-Q4_0.gguf?download=true"
    llm_url_mmproj = "https://huggingface.co/LiquidAI/LFM2-VL-450M-GGUF/resolve/main/mmproj-LFM2-VL-450M-Q8_0.gguf?download=true"



    # Modèles à vérifier et télécharger
    models_to_check: List[Tuple[str, str, str, str, bool]] = [
        # VOSK - Anglais (répertoire)
        (config['models']['stt_vosk']['en'],
         "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
         "zip",
         "VOSK English STT",
         True),
        
        # VOSK - Français (répertoire)
        (config['models']['stt_vosk']['fr'],
         "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
         "zip",
         "VOSK French STT",
         True),
        
        # Piper TTS - Français (fichier .onnx)
        (config['models']['tts_piper']['fr_upmc'],
         "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx?download=true",
         "file",
         "Piper TTS French",
         False),
        
        # Piper TTS - Français Config (fichier .json)
        (str(config['models']['tts_piper']['fr_upmc']) + ".json",
         "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx.json?download=true",
         "file",
         "Piper TTS French Config",
         False),

        # LLM text LFM2 8B A1B
        (config['models']['llm']['lfm_8b'],
         llm_url,
         "file",
         "LLM (mode CI léger)" if args.ci_mode else "LLM LFM2-8B",
         False),

        # LLM vision
        (config['models']['llm']['LFM2-VL-450M-Q4'],
         llm_url_vision,
         "file",
         "LFM2-VL 450M Vision",
         False),

        # LLM vision mmproj
        (config['models']['llm']['mmproj-LFM2-VL-450M-Q8'],
         llm_url_mmproj,
         "file",
         "LFM2-VL MMProj",
         False),

        # Llama.cpp server (fichier .exe)
        (config['executables']['llama_server']['path'],
         "https://github.com/ggml-org/llama.cpp/releases/download/b6987/llama-b6987-bin-win-cpu-x64.zip",
         "zip",
         "Llama.cpp Server",
         False),
    ]
    
    missing_count = 0
    downloaded_count = 0
    
    # Vérifie chaque modèle
    print("Vérification des modèles...")
    print("-" * 70)
    
    for model_path, url, dl_type, description, is_directory in models_to_check:
        print(f"\n🔍 Vérification: {description}")
        print(f"   Chemin attendu: {model_path}")
        
        if path_exists(model_path):
            item_type = "📁" if is_directory else "📄"
            print(f"   {item_type} Trouvé")
            continue
        
        print(f"   ❌ Manquant")
        missing_count += 1
        
        # Téléchargement
        temp_dir = tempfile.mkdtemp(prefix="robot_model_")
        temp_path = Path(temp_dir)
        
        try:
            if dl_type == "file":
                dest = PROJECT_ROOT / model_path
                success = download_file(url, dest, description)
                
            elif dl_type == "zip":
                zip_name = url.split('/')[-1].split('?')[0]
                zip_dest = temp_path / zip_name
                
                success = download_file(url, zip_dest, description)
                
                if success:
                    extract_dir = PROJECT_ROOT / Path(model_path).parent
                    expected_item = Path(model_path).name
                    
                    success = extract_zip(zip_dest, extract_dir, expected_item, is_directory)
            
            if success:
                downloaded_count += 1
                print(f"   ✅ Installation terminée avec succès")
            else:
                print(f"   ❌ Échec de l'installation")
                
        except Exception as e:
            print(f"   ❌ Erreur inattendue: {e}")
        finally:
            # Nettoyage
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)
    
    # Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"Modèles vérifiés: {len(models_to_check)}")
    print(f"Modèles manquants: {missing_count}")
    print(f"Modèles téléchargés: {downloaded_count}")
    
    if missing_count == downloaded_count:
        print("\n🎉 Tous les modèles sont maintenant disponibles!")
        sys.exit(0)
    elif downloaded_count > 0:
        print(f"\n⚠️  {downloaded_count}/{missing_count} modèles téléchargés.")
        print("   Vérifiez les erreurs ci-dessus pour les modèles manquants.")
        sys.exit(1)
    else:
        print("\n❌ Aucun modèle n'a pu être téléchargé.")
        sys.exit(1)

if __name__ == "__main__":
    main()