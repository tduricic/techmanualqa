"""
Unified LLM client that supports multiple backends (Google, OpenAI, LiteLLM).
"""
import os
import logging
from typing import Optional
from dataclasses import dataclass

try:
    import google.generativeai as genai
    import google.api_core.exceptions
    GOOGLE_AVAILABLE = True
except ImportError:
    genai = None
    GOOGLE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False


@dataclass
class LLMConfig:
    """Configuration for an LLM backend."""
    backend: str  # 'google' | 'openai' | 'litellm'
    model_name: str
    temperature: float = 0.0


class UnifiedLLMClient:
    """
    Wrapper class that provides a consistent interface across different LLM backends.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.backend = config.backend.lower()
        self.model_name = config.model_name
        self.temperature = config.temperature

        # Initialize the appropriate client
        if self.backend == "google":
            if not GOOGLE_AVAILABLE:
                raise ImportError("google-generativeai not installed. Install with: pip install google-generativeai")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model_name)
            logging.info(f"Initialized Google client with model: {self.model_name}")

        elif self.backend == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not installed. Install with: pip install openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = openai.OpenAI(api_key=api_key)
            logging.info(f"Initialized OpenAI client with model: {self.model_name}")

        elif self.backend == "litellm":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai library required for LiteLLM (uses OpenAI-compatible interface). Install with: pip install openai")
            api_key = os.getenv("LITELLM_API_KEY")
            endpoint = os.getenv("LITELLM_ENDPOINT", "https://cody.ib-inet.com/")
            if not api_key:
                raise ValueError("LITELLM_API_KEY not found in environment")
            # LiteLLM uses OpenAI-compatible interface
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=endpoint
            )
            logging.info(f"Initialized LiteLLM client with model: {self.model_name} at endpoint: {endpoint}")
        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Must be 'google', 'openai', or 'litellm'")

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate text from a prompt. Returns the generated text string.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Generated text as a string
        """
        try:
            if self.backend == "google":
                generation_config = {"temperature": self.temperature}
                if max_tokens:
                    generation_config["max_output_tokens"] = max_tokens

                response = self.client.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                # Extract text from Google response
                if response.candidates:
                    if response.candidates[0].content and response.candidates[0].content.parts:
                        return response.candidates[0].content.parts[0].text
                elif hasattr(response, 'text'):
                    return response.text
                return ""


            elif self.backend in ["openai", "litellm"]:
                kwargs = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "n": 1
                }
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                response = self.client.chat.completions.create(**kwargs)
                # Better error checking
                if not hasattr(response, 'choices') or not response.choices:
                    logging.error(f"No choices in response from {self.backend}")
                    return ""
                choice = response.choices[0]
                # Check finish_reason for issues
                if hasattr(choice, 'finish_reason'):
                    if choice.finish_reason == "length":
                        logging.warning(
                            f"Response truncated due to max_tokens limit ({max_tokens}). Consider increasing max_tokens.")
                    elif choice.finish_reason == "content_filter":
                        logging.warning("Response blocked by content filter")
                        return ""
                    elif choice.finish_reason != "stop":
                        logging.warning(f"Unusual finish_reason: {choice.finish_reason}")
                if choice.message and choice.message.content:
                    return choice.message.content.strip()
                else:
                    logging.error(f"No content in message. Choice: {choice}")
                    return ""
        except Exception as e:
            logging.error(f"Error in {self.backend} generate: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text. Returns -1 if not supported by backend.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens, or -1 if not supported
        """
        try:
            if self.backend == "google":
                response = self.client.count_tokens(text)
                return response.total_tokens
            elif self.backend in ["openai", "litellm"]:
                # Use tiktoken for OpenAI-compatible models
                try:
                    import tiktoken
                    # Use cl100k_base encoding (used by GPT-4, GPT-3.5-turbo, etc.)
                    encoding = tiktoken.get_encoding("cl100k_base")
                    tokens = encoding.encode(text)
                    return len(tokens)
                except ImportError:
                    logging.debug("tiktoken not installed. Install with: pip install tiktoken")
                    return -1
                except Exception as e:
                    logging.debug(f"Error counting tokens with tiktoken: {e}")
                    return -1
            else:
                logging.debug(f"Token counting not implemented for {self.backend}")
                return -1
        except Exception as e:
            logging.warning(f"Error counting tokens: {e}")
            return -1

    def is_rate_limit_error(self, exception: Exception) -> bool:
        """
        Check if an exception is a rate limit error.

        Args:
            exception: The exception to check

        Returns:
            True if it's a rate limit error, False otherwise
        """
        if self.backend == "google":
            if GOOGLE_AVAILABLE:
                return isinstance(exception, google.api_core.exceptions.ResourceExhausted)
            return False
        elif self.backend in ["openai", "litellm"]:
            if OPENAI_AVAILABLE:
                return isinstance(exception, openai.RateLimitError)
            return False
        return False


def create_llm_client(config_dict: dict) -> UnifiedLLMClient:
    """
    Factory function to create an LLM client from a config dictionary.

    Args:
        config_dict: Dictionary with 'backend', 'model_name', and optional 'temperature'

    Returns:
        UnifiedLLMClient instance

    Raises:
        KeyError: If required config keys are missing
        ValueError: If backend is invalid or API keys are missing
    """
    required_keys = ['backend', 'model_name']
    for key in required_keys:
        if key not in config_dict:
            raise KeyError(f"Missing required config key: '{key}'")

    config = LLMConfig(
        backend=config_dict['backend'],
        model_name=config_dict['model_name'],
        temperature=config_dict.get('temperature', 0.0)
    )
    return UnifiedLLMClient(config)