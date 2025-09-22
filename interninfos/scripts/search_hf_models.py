import requests

def search_hf_models(query):
    url = f"https://huggingface.co/api/models?search={query}"
    response = requests.get(url)
    if response.status_code == 200:
        models = response.json()
        for model in models:
            print(f"Model ID: {model.get('modelId')}")
            print(f"Tags: {model.get('tags')}")
            print("-" * 40)
    else:
        print(f"Failed to fetch models. Status code: {response.status_code}")

if __name__ == "__main__":
    search_hf_models("sarcasm")
