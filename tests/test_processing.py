# tests/test_processing.py

import sys
import pytest
import yaml
import os
import numpy as np
import subprocess
import time
import tempfile
import requests
from pathlib import Path
from PIL import Image

# Importer les classes et fonctions depuis votre code source
from src.data_acquisition.vosk_function import VoskRecognizer
from src.data_acquisition.mtcnn_function import detect_faces
from src.final_interaction.tts_piper import PiperTTS
from src.processing.chat import get_llm_response, get_llm_response_vision
from src.processing.function import choose_tool

# --- Fixtures Pytest ---

@pytest.fixture(scope="session")
def config():
    """Charge la configuration depuis config.yaml."""
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)


#################
#### SERVEUR ####
#################


@pytest.fixture(scope="session")
def run_llama_server(config):
    """
    Fixture pour démarrer le serveur LLM Texte.
    Compatible Windows/Linux et nouvelle config.
    """
    if not config['testing']['run_integration_tests']:
        print("\nSkipping llama-server startup.")
        yield None
        return

    # 1. Récupération du chemin de l'exécutable selon l'OS
    is_win = sys.platform == "win32"
    platform_key = 'win' if is_win else 'linux'
    
    try:
        exe_path_str = config['executables']['llama_server'][platform_key]
        server_path = Path(exe_path_str).resolve()
    except KeyError:
        pytest.fail(f"Config invalide pour l'OS {platform_key}", pytrace=False)

    if not server_path.exists():
        pytest.fail(f"L'exécutable du serveur n'a pas été trouvé : {server_path}", pytrace=False)

    # 2. Préparation de la commande
    model_path = Path(config['models']['llm']['lfm_8b'])
    args_template = config['executables']['llama_server']['args']
    args_str = args_template.format(model_path=model_path)
    
    command = [str(server_path)] + args_str.split()
    print(f"\n🚀 Démarrage Serveur Texte : {' '.join(command)}")

    # 3. Gestion des flags Windows uniquement
    creation_flags = 0
    if is_win:
        creation_flags = subprocess.CREATE_NO_WINDOW

    server_process = None
    try:
        server_process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            creationflags=creation_flags # Compatible Linux maintenant (0)
        )
        
        # 4. Attente active
        target_url = config['llm_server']['url']
        start_time = time.time()
        server_ready = False
        timeout = 60

        print(f"⏳ Attente connexion sur {target_url}...")
        while time.time() - start_time < timeout:
            if server_process.poll() is not None:
                stderr = server_process.stderr.read().decode('utf-8', errors='ignore')
                pytest.fail(f"Crash au démarrage:\n{stderr}", pytrace=False)
            
            try:
                requests.get(target_url, timeout=1)
                server_ready = True
                break
            except requests.exceptions.RequestException:
                time.sleep(1)
        
        if not server_ready:
            server_process.terminate()
            pytest.fail(f"Timeout sur {target_url}")
        
        print("✅ Serveur Texte prêt.")
        yield server_process
        
    finally:
        if server_process:
            print("🛑 Arrêt Serveur Texte.")
            server_process.terminate()
            server_process.wait()


@pytest.fixture(scope="session")
def run_llama_server_vision(config):
    """
    Fixture pour démarrer le serveur Vision.
    Compatible Windows/Linux et nouvelle config.
    """
    if not config['testing']['run_integration_tests']:
        yield None
        return

    # 1. Chemin Exécutable
    is_win = sys.platform == "win32"
    platform_key = 'win' if is_win else 'linux'
    
    exe_path_str = config['executables']['llama_server_vision'][platform_key]
    server_path = Path(exe_path_str).resolve()
    
    # 2. Modèles
    model_path = Path(config['models']['llm']['LFM2-VL-450M-Q4']).resolve()
    model_path_mmproj = Path(config['models']['llm']['mmproj-LFM2-VL-450M-Q8']).resolve()

    if not model_path.exists() or not model_path_mmproj.exists():
        pytest.fail(f"Modèles Vision manquants.\nCheck: {model_path}", pytrace=False)

    # 3. Commande
    args_template = config['executables']['llama_server_vision']['args']
    args_str = args_template.format(
        model_path=model_path, 
        model_path_mmproj=model_path_mmproj
    )
    
    command = [str(server_path)] + args_str.split()
    print(f"\n🚀 Démarrage Serveur Vision : {' '.join(command)}")

    # 4. Flags
    creation_flags = 0
    if is_win:
        creation_flags = subprocess.CREATE_NO_WINDOW

    # Fichiers logs temporaires
    stdout_file = tempfile.TemporaryFile()
    stderr_file = tempfile.TemporaryFile()

    server_process = subprocess.Popen(
        command, 
        stdout=stdout_file, 
        stderr=stderr_file,
        creationflags=creation_flags
    )
    
    server_url_root = "http://localhost:8088/health" # Endpoint santé souvent dispo
    # Fallback si health n'existe pas sur cette version de llama.cpp: racine
    
    start_time = time.time()
    ready = False
    timeout = 180 # Vision est lourd

    print(f"⏳ Attente connexion Vision (Port 8088)...")
    while time.time() - start_time < timeout:
        if server_process.poll() is not None:
            break
        try:
            # On essaye de taper la racine, ça renverra 404 ou 200, mais ça prouve que le serveur est up
            requests.get("http://localhost:8088/", timeout=1)
            ready = True
            break
        except requests.exceptions.RequestException:
            time.sleep(2)
    
    if not ready:
        stdout_file.seek(0); stderr_file.seek(0)
        out = stdout_file.read().decode('utf-8', errors='replace')
        err = stderr_file.read().decode('utf-8', errors='replace')
        pytest.fail(f"❌ Serveur Vision échec.\nSTDERR: {err}", pytrace=False)

    print("✅ Serveur Vision prêt.")
    yield server_process
        
    if server_process.poll() is None:
        server_process.terminate()
        server_process.wait()
    stdout_file.close()
    stderr_file.close()



@pytest.fixture(scope="module")
def setup_output_dir(config):
    """Crée le dossier de sortie pour les tests et le nettoie après."""
    output_dir = Path(config['testing']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    yield str(output_dir)
    # Nettoyage
    for f in output_dir.iterdir():
        try:
            f.unlink()
        except PermissionError:
            pass # Parfois windows bloque les fichiers temporairement
    try:
        output_dir.rmdir()
    except:
        pass


##############
#### TEST ####
##############

# --- Tests Unitaires (rapides) ---

def test_vosk_initialization(config):
    """Teste l'initialisation de Vosk avec le modèle français."""
    model_path = Path(config['models']['stt_vosk']['fr'])
    assert model_path.is_dir(), f"Le dossier du modèle Vosk n'existe pas : {model_path}"
    try:
        VoskRecognizer(model_path=str(model_path))
    except Exception as e:
        pytest.fail(f"L'initialisation de Vosk a échoué : {e}")

def test_mtcnn_face_detection():
    """Teste la détection de visages avec une image factice."""
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        faces = detect_faces(dummy_image)
        assert isinstance(faces, list)
    except Exception as e:
        pytest.fail(f"La détection de visage MTCNN a échoué : {e}")

def test_piper_tts_synthesis(config, setup_output_dir):
    """Teste la synthèse vocale Piper TTS."""
    model_path = Path(config['models']['tts_piper']['fr_upmc'])
    config_path = Path(str(model_path) + ".json")
    output_wav_path = Path(setup_output_dir) / "test_synthesis.wav"
    
    assert model_path.is_file(), f"Le fichier modèle Piper n'existe pas : {model_path}"
    assert config_path.is_file(), f"Le fichier config Piper n'existe pas : {config_path}"
    
    try:
        tts = PiperTTS(model_path=str(model_path))
        tts.synthesize("Ceci est un test.", str(output_wav_path), speaker_id=0)
        assert output_wav_path.exists()
        assert output_wav_path.stat().st_size > 0
    except Exception as e:
        pytest.fail(f"La synthèse vocale Piper a échoué : {e}")

def test_function_chooser_mocked(mocker, config):
    """Teste le choix de fonction avec un LLM mocké (rapide)."""
    mock_response = mocker.Mock()
    
    # 1. Adaptation de la réponse simulée : 
    # Le LLM renvoie maintenant juste le mot clé (souvent avec un espace avant)
    mock_response.json.return_value = {'content': ' get_time'}
    
    # 2. Important : On doit simuler le code 200, car ton code vérifie le 503
    mock_response.status_code = 200 
    mock_response.raise_for_status.return_value = None
    
    mocker.patch('requests.post', return_value=mock_response)
    
    server_config = config['llm_server']
    
    # 3. On pose une question liée au temps pour être cohérent
    chosen_tool = choose_tool("Quelle heure est-il ?", server_config['url'], server_config['headers'])
    
    # 4. On vérifie que le nettoyage (.strip()) a bien fonctionné
    assert chosen_tool == "get_time"

def test_chat_response_mocked(mocker, config):
    """Teste une réponse de chat avec un LLM mocké (rapide)."""
    mock_response = mocker.Mock()
    mock_response.json.return_value = {'content': 'Bonjour !'}
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.post', return_value=mock_response)
    server_config = config['llm_server']
    response = get_llm_response([{"role": "user", "content": "Salut"}], server_config['url'], server_config['headers'])
    assert response == "Bonjour !"

# --- Tests d'Intégration (lents) ---

@pytest.mark.skipif(
    not yaml.safe_load(open("config/config.yaml", encoding='utf-8'))['testing']['run_integration_tests'],
    reason="Intégration désactivée"
)
def test_vision_response_integration(config, run_llama_server_vision):
    """
    Teste la réponse vision en simulant main.py
    """
    assert run_llama_server_vision is not None

    # Création d'une image temporaire
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img = Image.new('RGB', (100, 100), color='red')
        img.save(tmp.name)
        tmp_path = tmp.name

    try:
        url = config['llm_server_vision']['url']
        response = get_llm_response_vision(
            server_url=url,
            image_path=tmp_path,
            prompt="Describe this image in one word."
        )
        
        print(f"\nRéponse Vision: {response}")
        # On check si la réponse est une string non vide (le modèle peut halluciner mais doit répondre)
        assert isinstance(response, str) and len(response) > 0

    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

def test_function_chooser_integration(config, run_llama_server):
    """Teste le choix de fonction avec le VRAI serveur LLM."""
    assert run_llama_server is not None, "Le serveur LLM n'a pas été démarré."
    
    server_config = config['llm_server']
    user_query = "Quelle heure est-il ?"
    
    try:
        chosen_tool = choose_tool(user_query, server_config['url'], server_config['headers'])
        print(f"Réponse du LLM (choix de fonction) : '{chosen_tool}'")
        # On assouplit l'assertion car les LLM quantifiés peuvent varier légèrement
        assert "get_time" in chosen_tool or "time" in chosen_tool
    except requests.exceptions.ConnectionError as e:
        pytest.fail(f"Échec de la connexion au serveur LLM local. Erreur : {e}")

def test_chat_response_integration(config, run_llama_server):
    """Teste une réponse de chat simple avec le VRAI serveur LLM."""
    assert run_llama_server is not None, "Le serveur LLM n'a pas été démarré."

    server_config = config['llm_server']
    history = [{"role": "user", "content": "Réponds juste 'Bonjour'."}]
    
    try:
        response = get_llm_response(history, server_config['url'], server_config['headers'])
        print(f"Réponse du LLM (chat) : '{response}'")
        assert isinstance(response, str)
        assert len(response) > 0
    except requests.exceptions.ConnectionError as e:
        pytest.fail(f"Échec de la connexion au serveur LLM local. Erreur : {e}")