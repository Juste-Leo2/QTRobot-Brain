#!/usr/bin/env python3
"""
download_model.py - Script de vérification et téléchargement des modèles IA
Supporte Windows et Linux, avec mode CI pour modèle léger.
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

def download_file(url: str, destination: Path, desc: str = "") -> bool:
    """
    Télécharge un fichier avec curl (compatible Win/Linux)
    """
    print(f"📥 Téléchargement {desc}...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Adaptation commande curl selon OS
    curl_cmd = 'curl'
    if sys.platform == "win32" and shutil.which('curl.exe'):
        curl_cmd = 'curl.exe'

    cmd = [
        curl_cmd, '-L', '-#', '-o', str(destination),
        '--connect-timeout', '30',
        '--retry', '3',
        '--retry-delay', '5',
        url
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Erreur lors du téléchargement {desc}")
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def extract_zip(zip_path: Path, extract_to: Path, expected_item: str, is_directory: bool) -> bool:
    """
    Extrait une archive zip avec gestion spécifique Linux pour les binaires
    """
    print(f"📦 Extraction de {zip_path.name}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Logique spéciale pour llama.cpp linux (souvent dans build/bin/)
            bin_prefix = "build/bin/"
            llama_linux_files = [f for f in zip_ref.namelist() if f.startswith(bin_prefix)]
            
            if llama_linux_files and sys.platform == "linux":
                found_in_zip = False
                for file_info in llama_linux_files:
                    file_name = file_info[len(bin_prefix):]
                    if file_name and file_name == expected_item:
                        source = zip_ref.open(file_info)
                        target_path = extract_to / expected_item
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        os.chmod(target_path, 0o755)
                        found_in_zip = True
                
                if found_in_zip:
                    print(f"   ✅ Binaire Linux extrait : {extract_to / expected_item}")
                    return True

            # Logique standard
            zip_ref.extractall(extract_to)

        # Vérification finale
        full_path = extract_to / expected_item
        if is_directory:
            if full_path.exists() and full_path.is_dir(): return True
        else:
            if full_path.exists() and full_path.is_file(): return True
            found = list(extract_to.rglob(expected_item))
            if found:
                shutil.move(str(found[0]), str(full_path))
                return True
        
        print(f"❌ Échec extraction : {expected_item} non trouvé.")
        return False

    except Exception as e:
        print(f"❌ Erreur extraction: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ci-mode', action='store_true', 
                       help='Utilise le modèle LLM léger pour CI')
    args = parser.parse_args()

    print("=" * 70)
    print(f"VÉRIFICATION ET TÉLÉCHARGEMENT DES MODÈLES IA [{sys.platform.upper()}]")
    print("=" * 70)

    config = load_config()
    is_linux = sys.platform == "linux"
    
    # 1. URLs des modèles
    
    # -- LLM Textuelle (Standard vs CI) --
    llm_text_url = "https://huggingface.co/unsloth/LFM2-8B-A1B-GGUF/resolve/main/LFM2-8B-A1B-UD-Q3_K_XL.gguf?download=true"
    
    if args.ci_mode:
        print("🚀 Mode CI activé - Utilisation du modèle LLM léger (350M)")
        # C'est ici qu'on remplace par le modèle texte 350M
        llm_text_url = "https://huggingface.co/LiquidAI/LFM2-350M-GGUF/resolve/main/LFM2-350M-Q4_K_M.gguf?download=true"

    # -- LLM Vision --
    llm_vision_url = "https://huggingface.co/LiquidAI/LFM2-VL-450M-GGUF/resolve/main/LFM2-VL-450M-Q4_0.gguf?download=true"
    llm_mmproj_url = "https://huggingface.co/LiquidAI/LFM2-VL-450M-GGUF/resolve/main/mmproj-LFM2-VL-450M-Q8_0.gguf?download=true"

    # -- Binaire Llama.cpp --
    if is_linux:
        llama_bin_url = "https://github.com/ggml-org/llama.cpp/releases/download/b6987/llama-b6987-bin-ubuntu-x64.zip"
        llama_exe_path = config['executables']['llama_server']['linux']
        llama_exe_name_in_zip = "llama-server" 
    else:
        llama_bin_url = "https://github.com/ggml-org/llama.cpp/releases/download/b6987/llama-b6987-bin-win-cpu-x64.zip"
        llama_exe_path = config['executables']['llama_server']['win']
        llama_exe_name_in_zip = "llama-server.exe"

    # Liste des tâches
    models_to_check: List[Tuple[str, str, str, str, bool]] = [
        # VOSK
        (config['models']['stt_vosk']['fr'], "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip", "zip", "VOSK FR", True),
        
        # TTS Piper
        (config['models']['tts_piper']['fr_upmc'], "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx?download=true", "file", "Piper TTS Model", False),
        (str(config['models']['tts_piper']['fr_upmc']) + ".json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx.json?download=true", "file", "Piper TTS Config", False),
        
        # LLM Text (Va télécharger le 350M mais le sauvegarder à la place du 8B si --ci-mode)
        (config['models']['llm']['lfm_8b'], llm_text_url, "file", "LLM Text Main", False),

        # LLM Vision
        (config['models']['llm']['LFM2-VL-450M-Q4'], llm_vision_url, "file", "LLM Vision Base", False),
        (config['models']['llm']['mmproj-LFM2-VL-450M-Q8'], llm_mmproj_url, "file", "LLM Vision Projector", False),

        # Llama Server Binary
        (llama_exe_path, llama_bin_url, "zip", "Llama Server Binary", False),
    ]

    # Exécution
    print(f"--- Vérification des {len(models_to_check)} fichiers requis ---")
    
    for model_path, url, dl_type, desc, is_dir in models_to_check:
        full_dest_path = PROJECT_ROOT / model_path
        
        if full_dest_path.exists():
            print(f"✅ {desc} présent.")
            # Attention : Si le fichier existe déjà (ex: le gros modèle), le script ne le remplace pas.
            # Pour switcher entre CI et Normal localement, il faut supprimer le fichier modèle manuellement.
            continue
            
        temp_dir = tempfile.mkdtemp()
        try:
            if dl_type == "file":
                download_file(url, full_dest_path, desc)
            else:
                zip_dest = Path(temp_dir) / "temp.zip"
                if download_file(url, zip_dest, desc):
                    extract_dir = PROJECT_ROOT / Path(model_path).parent
                    
                    if "Llama Server" in desc:
                        item_to_extract = llama_exe_name_in_zip
                    else:
                        item_to_extract = Path(model_path).name
                        
                    extract_zip(zip_dest, extract_dir, item_to_extract, is_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n🎉 Vérification terminée.")

if __name__ == "__main__":
    main()