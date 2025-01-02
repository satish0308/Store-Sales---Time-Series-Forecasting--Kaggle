import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from Exception import CustomException

import sys

# Server details
SERVERS = ["http://192.168.49.2:30656"]

# Paths to data
TRAIN_FILE = "df_hol_s.csv"
TEST_FILE = "test_data_hol_s.csv"
RESULT_DIR = "result/"  # Directory to save consolidated results

def split_data(train_file, test_file, servers):
    """
    Split unique IDs across servers.
    """
    # Read data
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # Find unique IDs
    unique_ids = pd.concat([train['default_rank'], test['default_rank']]).unique()
    unique_ids.sort()

    # Split IDs for servers
    chunk_size = len(unique_ids) // len(servers)
    id_chunks = [unique_ids[i:i + chunk_size] for i in range(0, len(unique_ids), chunk_size)]

    # Ensure all IDs are covered
    if len(id_chunks) > len(servers):
        id_chunks[-2].extend(id_chunks[-1])
        id_chunks.pop()

    return id_chunks

# def send_to_server(server, ids, train_file, test_file):
#     """
#     Send the task to a specific server.
#     """
#     url = f"http://{server}/process"
#     payload = {
#         "ids": ids.tolist(),
#         "train_file": train_file,
#         "test_file": test_file,
#     }

#     try:
#         response = requests.post(url, json=payload, timeout=60)
#         response.raise_for_status()
#         return "Server {server} completed: {response.json()}"
#     except requests.RequestException as e:
#         return f"Server {server} failed: {e}"

def send_to_server(server, ids, train_file, test_file):
    """
    Send the task to a specific server.
    """
    # Ensure the server URL doesn't have a redundant 'http://'
    if not server.startswith("http://"):
        server = f"http://{server}"
    
    url = f"{server}/process"
    payload = {
        "ids": ids.tolist(),
        "train_file": train_file,
        "test_file": test_file,
    }
    print(payload)

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return f"Server {server} completed: {response.json()}"
    except requests.RequestException as e:
        return f"Server {server} failed: {e}"


def distribute_computation():
    """
    Distribute computation across servers.
    """
    id_chunks = split_data(TRAIN_FILE, TEST_FILE, SERVERS)

    with ThreadPoolExecutor() as executor:
        futures = []
        for server, ids in zip(SERVERS, id_chunks):
            futures.append(executor.submit(send_to_server, server, ids, TRAIN_FILE, TEST_FILE))

        for future in futures:
            print(future.result())

if __name__ == "__main__":
    try:
        distribute_computation()
    except Exception as e:
        raise CustomException(e,sys)