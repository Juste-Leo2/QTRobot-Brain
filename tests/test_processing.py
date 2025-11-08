# tests/test_processing.py

import pytest
import yaml
import os
import numpy as np
import subprocess
import time
import requests
from pathlib import Path

# Importer les classes et fonctions depuis votre code source
from src.data_acquisition.vosk_function import VoskRecognizer
from src.data_acquisition.mtcnn_function import detect_faces
from src.final_interaction.tts_piper import PiperTTS
from src.processing.chat import get_llm_response
from src.processing.function import choose_tool

# --- Fixtures Pytest ---

@pytest.fixture(scope="session")
def config():
    """Charge la configuration depuis config.yaml."""
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="session")
def run_llama_server(config):
    """
    Fixture pour démarrer et arrêter le serveur llama-server.exe en arrière-plan.
    Compatible Windows avec gestion des processus.
    """
    if not config['testing']['run_integration_tests']:
        print("\nSkipping llama-server startup (integration tests disabled).")
        yield None
        return

    from pathlib import Path
    
    server_config = config['executables']['llama_server']
    server_path = Path(server_config['path'])
    
    # Vérifie que l'exécutable existe
    if not server_path.exists():
        # Cherche aussi dans le sous-dossier bin au cas où
        alt_path = server_path.parent / "bin" / "llama-server.exe"
        if alt_path.exists():
            server_path = alt_path
        else:
            pytest.fail(f"L'exécutable du serveur n'a pas été trouvé : {server_path}", pytrace=False)

    # Remplacer le placeholder dans les arguments
    model_path = Path(config['models']['llm']['lfm_8b_q4'])
    args_str = server_config['args'].format(model_path=model_path)
    
    # Construction de la commande Windows
    command = [str(server_path)] + args_str.split()

    print(f"\n🚀 Démarrage du serveur LLM avec la commande : {' '.join(command)}")
    
    server_process = None
    try:
        # Démarrage du processus Windows sans fenêtre console
        server_process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        print("⏳ Attente du démarrage du serveur...")
        time.sleep(15)  # Augmenté pour les gros modèles
        
        if server_process.poll() is not None:
            stderr_output = server_process.stderr.read().decode('utf-8', errors='ignore')
            pytest.fail(f"Le serveur n'a pas pu démarrer. Erreur:\n{stderr_output}", pytrace=False)
        
        print("✅ Serveur démarré.")
        yield server_process
        
    finally:
        if server_process:
            print("\n🛑 Arrêt du serveur LLM...")
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()
            print("✅ Serveur arrêté.")

@pytest.fixture(scope="module")
def setup_output_dir(config):
    """Crée le dossier de sortie pour les tests et le nettoie après."""
    from pathlib import Path
    output_dir = Path(config['testing']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    yield str(output_dir)
    # Nettoyage
    for f in output_dir.iterdir():
        f.unlink()
    output_dir.rmdir()

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
    mock_response.json.return_value = {'content': ' get_weather(city: Paris)'}
    mock_response.raise_for_status.return_value = None
    mocker.patch('requests.post', return_value=mock_response)
    server_config = config['llm_server']
    chosen_tool = choose_tool("Quel temps fait-il à Paris ?", server_config['url'], server_config['headers'])
    assert chosen_tool == "get_weather(city: Paris)"

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

integration_test = pytest.mark.skipif(
    not yaml.safe_load(open("config/config.yaml", encoding='utf-8'))['testing']['run_integration_tests'],
    reason="Les tests d'intégration sont désactivés dans config.yaml"
)

@integration_test
def test_function_chooser_integration(config, run_llama_server):
    """Teste le choix de fonction avec le VRAI serveur LLM."""
    assert run_llama_server is not None, "Le serveur LLM n'a pas été démarré."
    
    server_config = config['llm_server']
    user_query = "Quelle heure est-il ?"
    
    try:
        chosen_tool = choose_tool(user_query, server_config['url'], server_config['headers'])
        print(f"Réponse du LLM (choix de fonction) : '{chosen_tool}'")
        assert "get_time()" in chosen_tool
    except requests.exceptions.ConnectionError as e:
        pytest.fail(f"Échec de la connexion au serveur LLM local. Erreur : {e}")

@integration_test
def test_chat_response_integration(config, run_llama_server):
    """Teste une réponse de chat simple avec le VRAI serveur LLM."""
    assert run_llama_server is not None, "Le serveur LLM n'a pas été démarré."

    server_config = config['llm_server']
    history = [{"role": "user", "content": "Salut"}]
    
    try:
        response = get_llm_response(history, server_config['url'], server_config['headers'])
        print(f"Réponse du LLM (chat) : '{response}'")
        assert isinstance(response, str)
        assert len(response) > 0
    except requests.exceptions.ConnectionError as e:
        pytest.fail(f"Échec de la connexion au serveur LLM local. Erreur : {e}")