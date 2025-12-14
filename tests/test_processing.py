# tests/test_processing.py

import pytest
import os
import numpy as np
import subprocess
import time
import tempfile
import requests
import sys
from pathlib import Path
from PIL import Image

# Imports du code source
from src.data_acquisition.vosk_function import VoskRecognizer
from src.data_acquisition.mtcnn_function import detect_faces
from src.final_interaction.tts_piper import PiperTTS
from src.processing.chat import get_llm_response, get_llm_response_vision
from src.processing.function import choose_tool

# Import conditionnel API
try:
    from src.processing.api_google import GoogleGeminiHandler
except ImportError:
    GoogleGeminiHandler = None

# ==========================================
# FIXTURES SERVEURS (RESTENT ICI)
# ==========================================

@pytest.fixture(scope="session")
def run_llama_server(config):
    """Fixture serveur texte local."""
    if not config['testing']['run_integration_tests']:
        yield None
        return

    is_win = sys.platform == "win32"
    platform_key = 'win' if is_win else 'linux'
    
    try:
        exe_path_str = config['executables']['llama_server'][platform_key]
        server_path = Path(exe_path_str).resolve()
    except KeyError:
        pytest.fail(f"Config invalide pour l'OS {platform_key}", pytrace=False)

    if not server_path.exists():
        pytest.fail(f"Exécutable serveur introuvable: {server_path}", pytrace=False)

    model_path = Path(config['models']['llm']['lfm_8b'])
    args_str = config['executables']['llama_server']['args'].format(model_path=model_path)
    command = [str(server_path)] + args_str.split()
    
    print(f"\n🚀 Démarrage Serveur Texte...")

    my_env = os.environ.copy()
    if not is_win:
        exe_dir = str(server_path.parent)
        my_env["LD_LIBRARY_PATH"] = f"{exe_dir}:{my_env.get('LD_LIBRARY_PATH', '')}"

    creation_flags = subprocess.CREATE_NO_WINDOW if is_win else 0
    server_process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
        creationflags=creation_flags, env=my_env
    )
    
    target_url = config['llm_server']['url']
    start_time = time.time(); server_ready = False
    
    while time.time() - start_time < 60:
        if server_process.poll() is not None:
            pytest.fail("Crash démarrage serveur texte", pytrace=False)
        try:
            requests.get(target_url, timeout=1)
            server_ready = True
            break
        except requests.exceptions.RequestException:
            time.sleep(1)
    
    if not server_ready:
        server_process.terminate()
        pytest.fail(f"Timeout connexion {target_url}")
    
    yield server_process
    server_process.terminate()
    server_process.wait()

@pytest.fixture(scope="session")
def run_llama_server_vision(config):
    """Fixture serveur vision local."""
    if not config['testing']['run_integration_tests']:
        yield None
        return

    is_win = sys.platform == "win32"
    platform_key = 'win' if is_win else 'linux'
    exe_path_str = config['executables']['llama_server_vision'][platform_key]
    server_path = Path(exe_path_str).resolve()
    
    model_path = Path(config['models']['llm']['LFM2-VL-450M-Q4']).resolve()
    model_path_mmproj = Path(config['models']['llm']['mmproj-LFM2-VL-450M-Q8']).resolve()

    if not model_path.exists():
        pytest.fail("Modèles Vision manquants", pytrace=False)

    args_str = config['executables']['llama_server_vision']['args'].format(
        model_path=model_path, model_path_mmproj=model_path_mmproj
    )
    command = [str(server_path)] + args_str.split()
    print(f"\n🚀 Démarrage Serveur Vision...")

    creation_flags = subprocess.CREATE_NO_WINDOW if is_win else 0
    # On utilise des fichiers temp pour éviter de saturer le buffer PIPE
    stdout_file = tempfile.TemporaryFile(); stderr_file = tempfile.TemporaryFile()

    server_process = subprocess.Popen(
        command, stdout=stdout_file, stderr=stderr_file, creationflags=creation_flags
    )
    
    start_time = time.time(); ready = False
    while time.time() - start_time < 180:
        if server_process.poll() is not None: break
        try:
            requests.get("http://localhost:8088/", timeout=1)
            ready = True
            break
        except: time.sleep(2)
    
    if not ready:
        pytest.fail("Echec démarrage serveur Vision", pytrace=False)

    yield server_process
    server_process.terminate()
    server_process.wait()
    stdout_file.close(); stderr_file.close()

# ==========================================
# TESTS UNITAIRES
# ==========================================

def test_vosk_initialization(config):
    model_path = Path(config['models']['stt_vosk']['fr'])
    if not model_path.exists(): pytest.skip("Modèle Vosk absent")
    try: VoskRecognizer(model_path=str(model_path))
    except Exception as e: pytest.fail(f"Init Vosk: {e}")

def test_mtcnn_face_detection():
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        faces = detect_faces(dummy_image)
        assert isinstance(faces, list)
    except Exception as e: pytest.fail(f"MTCNN: {e}")

def test_piper_tts_synthesis(config, setup_output_dir):
    model_path = Path(config['models']['tts_piper']['fr_upmc'])
    if not model_path.exists(): pytest.skip("Modèle Piper absent")
    out = Path(setup_output_dir) / "test.wav"
    try:
        tts = PiperTTS(model_path=str(model_path))
        tts.synthesize("Test.", str(out), speaker_id=0)
        assert out.exists()
    except Exception as e: pytest.fail(f"TTS: {e}")

# ==========================================
# TEST API GOOGLE (NOUVEAU)
# ==========================================

def test_google_api_pipeline(api_key, config):
    """
    Test complet de l'API Google.
    Doit être SKIP si pas de clé.
    """
    # 1. Vérification PRIORITAIRE de la clé
    if not api_key:
        pytest.skip("⚠️ Pas de clé API fournie (--API 'KEY'). Test Google ignoré.")
    
    # 2. Vérification de l'import seulement après
    if GoogleGeminiHandler is None:
        pytest.fail("❌ Import de src.processing.api_google impossible.")

    print(f"\n☁️ [TEST] Validation API Google...")
    
    try:
        handler = GoogleGeminiHandler(api_key)
        
        # A. Test Router
        tool = handler.router_api("Quelle heure est-il ?")
        print(f"   👉 Router -> {tool}")
        assert tool in ["get_time", "None", "get_vision"]

        # B. Test Réponse Fusionnée
        fused = handler.generate_fused_response(
            "Fais coucou et montre un chat.", [], "None", config
        )
        print(f"   👉 Fusion -> Text: {len(fused['text'])} chars | Action: {fused['action']} | Display: {fused['display']}")

        assert "text" in fused
        assert "action" in fused
        assert "display" in fused
        assert len(fused["text"]) > 0

    except Exception as e:
        pytest.fail(f"❌ Erreur Test API: {e}")

# ==========================================
# TESTS LOCAUX (INTEGRATION)
# ==========================================

def test_local_integration_text(config, run_llama_server):
    if not run_llama_server: pytest.skip("Serveur Local non démarré")
    
    # Test Choose Tool
    res = choose_tool("Quelle heure est-il ?", config['llm_server']['url'], config['llm_server']['headers'])
    assert "time" in res or "get_time" in res
    
    # Test Chat
    res_chat = get_llm_response([{"role":"user", "content":"Salut"}], config['llm_server']['url'], config['llm_server']['headers'])
    assert len(res_chat) > 0

def test_local_integration_vision(config, run_llama_server_vision):
    if not run_llama_server_vision: pytest.skip("Serveur Vision non démarré")
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        Image.new('RGB', (100, 100), color='red').save(tmp.name)
        tmp_path = tmp.name
    try:
        res = get_llm_response_vision(config['llm_server_vision']['url'], tmp_path, "Describe")
        assert len(res) > 0
    finally:
        try: os.unlink(tmp_path)
        except: pass