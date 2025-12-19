"""Module for Local LLM client using Transformers.

Transformersを使用したローカルLLMクライアントのモジュール.
"""

import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from llm.client import BaseLLMClient
from llm.formatters import BasePromptFormatter, ChatTemplateFormatter, SimplePromptFormatter

logger = logging.getLogger(__name__)


class LocalLLMClient(BaseLLMClient):
    """Client for Local LLM using Transformers.

    Transformersを使用したローカルLLM用クライアント.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the Local LLM client.

        ローカルLLMクライアントを初期化する.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
        """
        self.model_path = str(config.get("local_model_path", ""))
        if not self.model_path:
             # If no local path, try to use model name as path (e.g. for HF hub)
             self.model_path = str(config.get("model", ""))
        
        self.temperature = float(config.get("temperature", 0.7))
        
        # Formatter selection strategy
        formatter_type = config.get("formatter", "chat_template")
        self.formatter: BasePromptFormatter
        if formatter_type == "simple":
            self.formatter = SimplePromptFormatter()
        else:
            self.formatter = ChatTemplateFormatter()
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        
        logger.info(f"Loading local model from {self.model_path} on {device}...")
        
        try:
            self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=device,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            )
            self.device = device
            logger.info("Local model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text using Local LLM.

        ローカルLLMを使用してテキストを生成する.

        Args:
            system_prompt (str): System prompt / システムプロンプト
            user_prompt (str): User prompt / ユーザプロンプト

        Returns:
            str: Generated text / 生成されたテキスト
        """
        try:
            # Delegate prompt formatting to the strategy
            prompt = self.formatter.format(system_prompt, user_prompt, self.tokenizer)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256, # Should be configurable
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-processing to extract response might depend on the formatter/model
            # For now, we try to be generic or rely on the fact that decode often includes the prompt
            # Ideally, the formatter could also handle extraction or we use a more robust way
            
            # Simple heuristic: if prompt is in generated text, remove it.
            # However, prompt string might differ from decoded tokens slightly.
            # A common approach is to decode only the new tokens:
            input_length = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True) # type: ignore[attr-defined]
            
            return str(response.strip())
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return ""
