# tests/test_processing.py

import pytest
import yaml
import os
import numpy as np
import subprocess
import time
import tempfile
import requests
import sys
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
    """
    if not config['testing']['run_integration_tests']:
        print("\nSkipping llama-server startup.")
        yield None
        return

    # 1. Chemin
    is_win = sys.platform == "win32"
    platform_key = 'win' if is_win else 'linux'
    
    try:
        exe_path_str = config['executables']['llama_server'][platform_key]
        server_path = Path(exe_path_str).resolve()
    except KeyError:
        pytest.fail(f"Config invalide pour l'OS {platform_key}", pytrace=False)

    if not server_path.exists():
        pytest.fail(f"L'exécutable du serveur n'a pas été trouvé : {server_path}", pytrace=False)

    # 2. Commande
    model_path = Path(config['models']['llm']['lfm_8b'])
    args_template = config['executables']['llama_server']['args']
    args_str = args_template.format(model_path=model_path)
    command = [str(server_path)] + args_str.split()
    print(f"\n🚀 Démarrage Serveur Texte : {' '.join(command)}")

    # 3. Environnement (Fix Linux Libs)
    my_env = os.environ.copy()
    if not is_win:
        exe_dir = str(server_path.parent)
        my_env["LD_LIBRARY_PATH"] = f"{exe_dir}:{my_env.get('LD_LIBRARY_PATH', '')}"

    creation_flags = subprocess.CREATE_NO_WINDOW if is_win else 0

    server_process = None
    try:
        server_process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            creationflags=creation_flags,
            env=my_env
        )
        
        # 4. Attente
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
            server_process.terminate()
            server_process.wait()


@pytest.fixture(scope="session")
def run_llama_server_vision(config):
    """
    Fixture pour démarrer le serveur Vision.
    """
    if not config['testing']['run_integration_tests']:
        yield None
        return

    # 1. Chemin
    is_win = sys.platform == "win32"
    platform_key = 'win' if is_win else 'linux'
    exe_path_str = config['executables']['llama_server_vision'][platform_key]
    server_path = Path(exe_path_str).resolve()
    
    # 2. Modèles
    model_path = Path(config['models']['llm']['LFM2-VL-450M-Q4']).resolve()
    model_path_mmproj = Path(config['models']['llm']['mmproj-LFM2-VL-450M-Q8']).resolve()

    if not model_path.exists():
        pytest.fail(f"Modèles Vision manquants.\nCheck: {model_path}", pytrace=False)

    # 3. Commande
    args_template = config['executables']['llama_server_vision']['args']
    args_str = args_template.format(
        model_path=model_path, 
        model_path_mmproj=model_path_mmproj
    )
    command = [str(server_path)] + args_str.split()
    print(f"\n🚀 Démarrage Serveur Vision : {' '.join(command)}")

    # 4. Environnement (Fix Linux Libs)
    my_env = os.environ.copy()
    if not is_win:
        exe_dir = str(server_path.parent)
        my_env["LD_LIBRARY_PATH"] = f"{exe_dir}:{my_env.get('LD_LIBRARY_PATH', '')}"

    creation_flags = subprocess.CREATE_NO_WINDOW if is_win else 0

    stdout_file = tempfile.TemporaryFile()
    stderr_file = tempfile.TemporaryFile()

    server_process = subprocess.Popen(
        command, 
        stdout=stdout_file, 
        stderr=stderr_file,
        creationflags=creation_flags,
        env=my_env
    )
    
    start_time = time.time()
    ready = False
    timeout = 180 

    print(f"⏳ Attente connexion Vision (Port 8088)...")
    while time.time() - start_time < timeout:
        if server_process.poll() is not None:
            break
        try:
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
    """Crée le dossier de sortie pour les tests."""
    output_dir = Path(config['testing']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    yield str(output_dir)
    try:
        for f in output_dir.iterdir():
            try: f.unlink()
            except: pass
        output_dir.rmdir()
    except: pass


# --- Tests Unitaires (inchangés) ---

def test_vosk_initialization(config):
    model_path = Path(config['models']['stt_vosk']['fr'])
    assert model_path.is_dir()
    try:
        VoskRecognizer(model_path=str(model_path))
    except Exception as e:
        pytest.fail(f"Init Vosk échec: {e}")

def test_mtcnn_face_detection():
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        faces = detect_faces(dummy_image)
        assert isinstance(faces, list)
    except Exception as e:
        pytest.fail(f"MTCNN échec: {e}")

def test_piper_tts_synthesis(config, setup_output_dir):
    model_path = Path(config['models']['tts_piper']['fr_upmc'])
    output_wav_path = Path(setup_output_dir) / "test_synthesis.wav"
    try:
        tts = PiperTTS(model_path=str(model_path))
        tts.synthesize("Test.", str(output_wav_path), speaker_id=0)
        assert output_wav_path.exists()
    except Exception as e:
        pytest.fail(f"TTS échec: {e}")

def test_function_chooser_mocked(mocker, config):
    mock_response = mocker.Mock()
    mock_response.json.return_value = {'content': ' get_time'}
    mock_response.status_code = 200 
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.post', return_value=mock_response)
    server_config = config['llm_server']
    chosen_tool = choose_tool("Quelle heure est-il ?", server_config['url'], server_config['headers'])
    assert chosen_tool == "get_time"

def test_chat_response_mocked(mocker, config):
    mock_response = mocker.Mock()
    mock_response.json.return_value = {'content': 'Bonjour !'}
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.post', return_value=mock_response)
    server_config = config['llm_server']
    response = get_llm_response([{"role": "user", "content": "Salut"}], server_config['url'], server_config['headers'])
    assert response == "Bonjour !"

# --- Tests Intégration (inchangés) ---

@pytest.mark.skipif(
    not yaml.safe_load(open("config/config.yaml", encoding='utf-8'))['testing']['run_integration_tests'],
    reason="Intégration désactivée"
)
def test_vision_response_integration(config, run_llama_server_vision):
    assert run_llama_server_vision is not None
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        Image.new('RGB', (100, 100), color='red').save(tmp.name)
        tmp_path = tmp.name
    try:
        url = config['llm_server_vision']['url']
        response = get_llm_response_vision(server_url=url, image_path=tmp_path, prompt="Describe.")
        assert isinstance(response, str) and len(response) > 0
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

def test_function_chooser_integration(config, run_llama_server):
    assert run_llama_server is not None
    server_config = config['llm_server']
    chosen_tool = choose_tool("Quelle heure est-il ?", server_config['url'], server_config['headers'])
    assert "get_time" in chosen_tool or "time" in chosen_tool

def test_chat_response_integration(config, run_llama_server):
    assert run_llama_server is not None
    server_config = config['llm_server']
    response = get_llm_response([{"role": "user", "content": "Bonjour."}], server_config['url'], server_config['headers'])
    assert len(response) > 0