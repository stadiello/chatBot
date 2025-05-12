import threading
import subprocess
import time
import sys
import os

# Ajout du répertoire racine au PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
os.environ["PYTHONPATH"] = project_root

def init_db():
    from src.rag.new_chromadb import rag_pipeline
    # Initialisation silencieuse de la base de données
    try:
        rag_pipeline("test")
    except Exception as e:
        print(f"Erreur lors de l'initialisation de la base de données : {e}")

def run_app():
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
    subprocess.run(["streamlit", "run", "src/streamapp.py"], env=env)
    
if __name__ == "__main__":
    try:
        db_thread = threading.Thread(target=init_db)
        streamlit_thread = threading.Thread(target=run_app)
        db_thread.start()
        time.sleep(5)
        streamlit_thread.start()
        db_thread.join()
        streamlit_thread.join()
    except KeyboardInterrupt:
        print("Interruption détectée. Arrêt du programme.")