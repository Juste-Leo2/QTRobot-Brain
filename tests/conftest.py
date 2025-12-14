# tests/conftest.py
import pytest
import yaml
from pathlib import Path

# --- 1. AJOUT DE L'OPTION CLI ---
def pytest_addoption(parser):
    """Permet à Pytest de reconnaître l'argument --api-key"""
    parser.addoption("--api-key", action="store", default=None, help="Clé API Google Gemini")

# --- 2. FIXTURES GLOBALES ---
@pytest.fixture
def api_key(request):
    """Récupère la valeur de --api-key"""
    return request.config.getoption("--api-key")

@pytest.fixture(scope="session")
def config():
    """Charge la configuration depuis config.yaml une seule fois."""
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

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