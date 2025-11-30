import requests
import json


class Models:

    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        self.base_url = "https://api.helmholtz-blablador.fz-juelich.de/v1"

    def get_model_data(self):
        response = requests.get(url=f"{self.base_url}/models",
                                headers=self.headers)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()["data"]

    def get_model_ids(self):
        model_data = self.get_model_data()
        return [model["id"] for model in model_data]


class ChatCompletions:

    def __init__(self,
                 api_key,
                 model,
                 temperature=0.7,
                 choices=1,
                 max_tokens=100,
                 user='default'):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.choices = choices
        self.max_tokens = max_tokens
        self.user = user
        self.headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.base_url = "https://api.helmholtz-blablador.fz-juelich.de/v1"

    def get_completion(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "n": self.choices,
            "max_tokens": self.max_tokens,
            "user": self.user
        }

        response = requests.post(
            url=f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload  # Use json parameter instead of data + json.dumps
        )

        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()


class Completions:

    def __init__(self,
                 api_key,
                 model,
                 temperature=0.7,
                 choices=1,
                 max_tokens=50,
                 user="default"):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.choices = choices
        self.max_tokens = max_tokens
        self.user = user
        self.headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.base_url = "https://api.helmholtz-blablador.fz-juelich.de/v1"

    def get_completion(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "n": self.choices,
            "max_tokens": self.max_tokens,
            "user": self.user
        }

        response = requests.post(url=f"{self.base_url}/completions",
                                 headers=self.headers,
                                 json=payload)

        response.raise_for_status()
        return response.json()


class TokenCount:

    def __init__(self, model, max_tokens=0):
        self.model = model
        self.max_tokens = max_tokens
        self.headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        self.base_url = "https://api.helmholtz-blablador.fz-juelich.de/v1/token_check"

    def count(self, prompts):
        try:
            iterator = iter(prompts)
        except TypeError:
            prompt_list = [{
                "model": self.model,
                "prompt": prompts,
                "max_tokens": self.max_tokens
            }]
        else:
            prompt_list = []
            for prompt in prompts:
                prompt_list.append({
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": self.max_tokens
                })

        payload = {"prompts": prompt_list}

        response = requests.post(url=self.base_url,
                                 headers=self.headers,
                                 json=payload)

        response.raise_for_status()
        return response.json()
