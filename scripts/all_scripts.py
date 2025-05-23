import subprocess

if __name__ == "__main__":
    subprocess.run(["python3", "data_script.py"])
    subprocess.run(["python3", "training_script.py"])
    subprocess.run(["python3", "inference_script.py"])
