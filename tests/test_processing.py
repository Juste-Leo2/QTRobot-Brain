# tests/test_processing.py

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
    Fixture pour démarrer et arrêter le serveur llama-server.exe en arrière-plan.
    Remplace le sleep par une vérification active de la connexion.
    """
    if not config['testing']['run_integration_tests']:
        print("\nSkipping llama-server startup (integration tests disabled).")
        yield None
        return

    server_config = config['executables']['llama_server']
    server_path = Path(server_config['path'])
    
    # Vérifie que l'exécutable existe
    if not server_path.exists():
        alt_path = server_path.parent / "bin" / "llama-server.exe"
        if alt_path.exists():
            server_path = alt_path
        else:
            pytest.fail(f"L'exécutable du serveur n'a pas été trouvé : {server_path}", pytrace=False)

    # Remplacer le placeholder dans les arguments
    model_path = Path(config['models']['llm']['lfm_8b'])
    args_str = server_config['args'].format(model_path=model_path)
    
    # Construction de la commande Windows
    command = [str(server_path)] + args_str.split()

    print(f"\n🚀 Démarrage du serveur LLM (Texte) avec la commande : {' '.join(command)}")
    
    server_process = None
    try:
        # Démarrage du processus Windows sans fenêtre console
        server_process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        # --- MODIFICATION: Gestion du Timeout intelligente (au lieu de sleep) ---
        print("⏳ Attente de la disponibilité du serveur Texte...")
        
        # On récupère l'URL cible depuis la config pour tester la connexion
        target_url = config['llm_server']['url']
        start_time = time.time()
        server_ready = False
        timeout = 60  # 60 secondes max pour charger le modèle

        while time.time() - start_time < timeout:
            # Vérifier si le processus a crashé immédiatement
            if server_process.poll() is not None:
                stderr_output = server_process.stderr.read().decode('utf-8', errors='ignore')
                pytest.fail(f"Le serveur s'est arrêté prématurément pendant le démarrage:\n{stderr_output}", pytrace=False)
            
            try:
                # On tente une requête simple. Même si on reçoit une 404 ou 405 (Method Not Allowed),
                # cela signifie que le serveur HTTP tourne.
                requests.get(target_url, timeout=1)
                server_ready = True
                break
            except requests.exceptions.RequestException:
                # Le serveur n'est pas encore prêt, on attend 1 seconde
                time.sleep(1)
        
        if not server_ready:
            server_process.terminate()
            pytest.fail(f"Timeout : Le serveur LLM n'a pas répondu après {timeout} secondes sur {target_url}.")
        
        print(f"✅ Serveur Texte prêt en {round(time.time() - start_time, 2)}s.")
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


@pytest.fixture(scope="session")
def run_llama_server_vision(config):
    """
    Fixture pour démarrer le serveur vision.
    Améliorée pour gérer les délais longs en CI/CD et logger les erreurs.
    """
    if not config['testing']['run_integration_tests']:
        yield None
        return

    server_config = config['executables']['llama_server_vision']
    server_path = Path(server_config['path'])
    
    if not server_path.exists():
         server_path = server_path.parent / "bin" / "llama-server.exe"
    
    model_path = Path(config['models']['llm']['LFM2-VL-450M-Q4']).resolve()
    model_path_mmproj = Path(config['models']['llm']['mmproj-LFM2-VL-450M-Q8']).resolve()
    
    # Vérification préalable de l'existence des modèles
    if not model_path.exists() or not model_path_mmproj.exists():
        pytest.fail(f"Modèles manquants sur le runner.\nModel: {model_path}\nProj: {model_path_mmproj}", pytrace=False)

    args_str = server_config['args'].format(
        model_path=model_path, 
        model_path_mmproj=model_path_mmproj
    )
    
    command = [str(server_path)] + args_str.split()
    
    # Détection si on est sur GitHub Actions ou un environnement CI lent
    is_ci = os.getenv('CI') or os.getenv('GITHUB_ACTIONS')
    timeout_duration = 180 if is_ci else 60  # 3 minutes en CI, 1 minute en local
    
    print(f"\n🚀 [Vision] Démarrage (Timeout set à {timeout_duration}s) : {' '.join(command)}")
    
    # On redirige stdout/stderr vers des fichiers temporaires pour éviter les blocages de buffer
    # et pour pouvoir les lire facilement en cas d'erreur.
    stdout_file = tempfile.TemporaryFile()
    stderr_file = tempfile.TemporaryFile()

    server_process = subprocess.Popen(
        command, 
        stdout=stdout_file, 
        stderr=stderr_file,
        creationflags=subprocess.CREATE_NO_WINDOW
    )
    
    server_url_root = "http://localhost:8088/"
    start_time = time.time()
    ready = False
    
    try:
        while time.time() - start_time < timeout_duration:
            # 1. Vérifier si le processus est mort
            if server_process.poll() is not None:
                break # Sort de la boucle pour traiter l'erreur

            # 2. Tenter la connexion
            try:
                requests.get(server_url_root, timeout=1)
                ready = True
                break
            except requests.exceptions.RequestException:
                time.sleep(2) # Attendre un peu plus entre les essais
        
        # --- GESTION DES ERREURS ET LOGS ---
        if not ready:
            # Lecture des logs pour le débogage
            stdout_file.seek(0)
            stderr_file.seek(0)
            out = stdout_file.read().decode('utf-8', errors='replace')
            err = stderr_file.read().decode('utf-8', errors='replace')
            
            return_code = server_process.poll()
            
            if return_code is not None:
                msg = f"❌ Le serveur Vision a crashé (Code: {return_code})."
            else:
                msg = f"❌ Timeout : Le serveur Vision ne répond pas après {timeout_duration}s."
                server_process.terminate()

            # On affiche les logs complets dans l'erreur pytest
            pytest.fail(f"{msg}\n\n--- STDOUT ---\n{out}\n\n--- STDERR ---\n{err}", pytrace=False)

        print(f"✅ Serveur Vision prêt en {round(time.time() - start_time, 2)}s.")
        yield server_process
        
    finally:
        # Nettoyage
        if server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_process.kill()
        
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