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
import tarfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# --- FIX WINDOWS ENCODING ---
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

def download_file(url: str, destination: Path, desc: str = "") -> bool:
    """
    Télécharge un fichier avec curl (compatible Win/Linux)
    """
    print(f"📥 Téléchargement {desc}...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    
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

def extract_archive(archive_path: Path, extract_to: Path, expected_item: str, is_directory: bool) -> bool:
    """
    Extrait une archive zip ou tar.gz. 
    Sur Linux, extrait TOUT le contenu du dossier binaire pour avoir les libs (.so).
    """
    print(f"📦 Extraction de {archive_path.name}...")
    
    try:
        bin_prefix = "build/bin/"
        found_binary = False
        
        # Logique spéciale pour llama.cpp linux (souvent dans build/bin/ ou similaire)
        # On extrait tout de toute façon car les libraries partagées sont nécessaires.
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                llama_linux_files = [f for f in zip_ref.namelist() if f.startswith(bin_prefix)]
                
                if llama_linux_files and sys.platform == "linux":
                    print("   -> Extraction complète des binaires et libs Linux (zip)...")
                    for file_info in llama_linux_files:
                        file_name = file_info[len(bin_prefix):]
                        if file_name: 
                            source = zip_ref.open(file_info)
                            target_path = extract_to / file_name
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(target_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
                            if file_name == expected_item:
                                os.chmod(target_path, 0o755)
                                found_binary = True
                    if found_binary:
                        print(f"   ✅ Binaire et Libs Linux extraits dans : {extract_to}")
                        return True
                else:
                    zip_ref.extractall(extract_to)
                    
        elif archive_path.name.endswith('.tar.gz'):
             with tarfile.open(archive_path, 'r:gz') as tar_ref:
                # Les tar.gz Linux llama.cpp ont souvent une structure différente selon les builds
                # Extrayons tout. On cherchera le binaire ensuite.
                print("   -> Extraction complète du tar.gz...")
                tar_ref.extractall(extract_to)

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
        
        print(f"❌ Échec extraction : {expected_item} non trouvé. Contenu extrait : {[p.name for p in extract_to.iterdir()]}")
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
    
    # URLs
    llm_text_url = "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-UD-Q4_K_XL.gguf?download=true"
    
    if args.ci_mode:
        print("🚀 Mode CI activé - Utilisation du modèle LLM léger")
        llm_text_url = "https://huggingface.co/unsloth/Qwen3.5-0.5B-GGUF/resolve/main/Qwen3.5-0.5B-Q4_K_M.gguf?download=true"

    if is_linux:
        llama_bin_url = "https://github.com/ggml-org/llama.cpp/releases/download/b8287/llama-b8287-bin-ubuntu-x64.tar.gz"
        llama_exe_path = config['executables']['llama_server']['linux']
        llama_exe_name_in_zip = "llama-server" 
    else:
        llama_bin_url = "https://github.com/ggml-org/llama.cpp/releases/download/b8287/llama-b8287-bin-win-cpu-x64.zip"
        llama_exe_path = config['executables']['llama_server']['win']
        llama_exe_name_in_zip = "llama-server.exe"

    models_to_check: List[Tuple[str, str, str, str, bool]] = [
        (config['models']['stt_vosk']['fr'], "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip", "zip", "VOSK FR", True),
        (config['models']['tts_piper']['fr_upmc'], "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx?download=true", "file", "Piper TTS Model", False),
        (str(config['models']['tts_piper']['fr_upmc']) + ".json", "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx.json?download=true", "file", "Piper TTS Config", False),
        (config['models']['llm']['qwen3.5_0_8b'], llm_text_url, "file", "LLM Main Model", False),
        (llama_exe_path, llama_bin_url, "archive", "Llama Server Binary", False),
    ]

    print(f"--- Vérification des {len(models_to_check)} fichiers requis ---")
    
    for model_path, url, dl_type, desc, is_dir in models_to_check:
        full_dest_path = PROJECT_ROOT / model_path
        
        if full_dest_path.exists():
            print(f"✅ {desc} présent.")
            continue
            
        temp_dir = tempfile.mkdtemp()
        try:
            if dl_type == "file":
                download_file(url, full_dest_path, desc)
            else:
                archive_name = "temp.zip" if "zip" in url else "temp.tar.gz"
                archive_dest = Path(temp_dir) / archive_name
                if download_file(url, archive_dest, desc):
                    extract_dir = PROJECT_ROOT / Path(model_path).parent
                    
                    if "Llama Server" in desc:
                        item_to_extract = llama_exe_name_in_zip
                    else:
                        item_to_extract = Path(model_path).name
                        
                    extract_archive(archive_dest, extract_dir, item_to_extract, is_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n🎉 Vérification terminée.")

if __name__ == "__main__":
    main()