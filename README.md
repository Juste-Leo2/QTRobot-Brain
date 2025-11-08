pytest -v


# Activer le mode CI avant de push
python src/download_model.py --ci-mode

# Lancer les tests comme GitHub Actions
pytest -v --tb=short

# Vérifier la taille du dossier models
Get-ChildItem -Recurse models\ | Measure-Object -Property Length -Sum
