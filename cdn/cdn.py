import os
import requests

def download_model(model):
    host = f"http://{os.getenv('CDN_HOST')}"
    path = f"{os.getenv('CDN_MODELS_PATH')}/{model}"
    auth = (os.getenv('CDN_USERNAME'), os.getenv('CDN_PASSWORD'))
    url = f"{host}/{path}"
    
    response = requests.get(url, auth=auth)
    if response.status_code != 200:
        raise ValueError(f"Failed to download model {model}: {response.status_code}")
    
    with open(model, 'wb') as f:
        f.write(response.content)
    
    return model

def upload_model(save_path, item_id):
    host = f"http://{os.getenv('CDN_HOST')}"
    path = f"{os.getenv('CDN_MODELS_PATH')}/prophet_bilstm_model_item_{item_id}.pth"
    auth = (os.getenv('CDN_USERNAME'), os.getenv('CDN_PASSWORD'))
    url = f"{host}/{path}"

    with open(save_path, 'rb') as f:
        headers = {"Content-Type": "application/octet-stream"}
        response = requests.put(url, headers=headers, data=f, auth=auth)
        response.raise_for_status()
        f.close()
    
    return path

def exists_model(item_id):
    exists = False
    host = f"http://{os.getenv('CDN_HOST')}"
    auth = (os.getenv('CDN_USERNAME'), os.getenv('CDN_PASSWORD'))

    path1 = f"{os.getenv('CDN_MODELS_PATH')}/prophet_bilstm_model_item_{item_id}.pth"
    path2 = f"{os.getenv('CDN_MODELS_PATH')}/lstm_model_item_{item_id}.pth"

    url1 = f"{host}/{path1}"
    url2 = f"{host}/{path2}"

    response1 = requests.head(url1, auth=auth)
    response2 = requests.head(url2, auth=auth)
    return response1.status_code == 200 or response2.status_code == 200