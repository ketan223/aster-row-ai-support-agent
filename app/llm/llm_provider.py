import os
import requests
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

class LLMProvider(ABC):
    # Sends a list of conversation messages to the LLM and returns the text response.
    @abstractmethod
    def chat(self, messages: list[dict], system_prompt: str = None) -> str:
        pass

class OllamaProvider(LLMProvider):
    # Initializes the Ollama provider with a model and an API base URL.
    def __init__(self, model: str = None, api_base: str = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        self.api_base = api_base or os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

    # Executes a chat request to a local Ollama service using its HTTP API.
    def chat(self, messages: list[dict], system_prompt: str = None) -> str:
        url = f"{self.api_base}/api/chat"
        
        payload_messages = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})
        payload_messages.extend(messages)
        
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "stream": False,
            "options": {
                "temperature": 0.0  # Set to 0.0 for deterministic evaluation answers
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
        except Exception as e:
            # Return an error message or raise
            raise RuntimeError(f"Ollama provider failed: {str(e)}")

class MistralProvider(LLMProvider):
    # Initializes the Mistral provider with an API key and model name.
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = model or os.getenv("MISTRAL_MODEL", "open-mistral-7b")

    # Executes a completion request against Mistral AI's standard REST API.
    def chat(self, messages: list[dict], system_prompt: str = None) -> str:
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set.")
            
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload_messages = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})
        payload_messages.extend(messages)
        
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": 0.0
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Mistral provider failed: {str(e)}")

# Factory function that creates and returns the configured LLMProvider subclass.
def get_llm_provider() -> LLMProvider:
    provider_name = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider_name == "mistral":
        return MistralProvider()
    else:
        return OllamaProvider()
