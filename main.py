import threading
import subprocess
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def init_db():
    subprocess.run(["python", "src/rag/new_chromadb.py"])

def run_app():
    subprocess.run(["streamlit", "run", "src/streamapp.py"])
    
if __name__ == "__main__":
    db_thread = threading.Thread(target=init_db)
    streamlit_thread = threading.Thread(target=run_app)
    db_thread.start()
    time.sleep(5)
    streamlit_thread.start()
    db_thread.join()
    streamlit_thread.join()