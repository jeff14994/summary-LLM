"""
Local LLM Module

This module handles interaction with the Ollama runtime for local LLM inference.
"""

import subprocess
import json
from typing import Optional, Dict, Any
from loguru import logger
import os
from dotenv import load_dotenv

class LocalLLM:
    def __init__(self):
        load_dotenv()
        self.model = os.getenv("OLLAMA_MODEL", "jcai/breeze-7b-32k-instruct-v1_0:f16")
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))  # Default 5 minutes
        
    def _check_ollama_installed(self) -> bool:
        """
        Check if Ollama is installed and running.
        
        Returns:
            bool: True if Ollama is installed and running, False otherwise
        """
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
            
    def _check_model_available(self) -> bool:
        """
        Check if the required model is available locally.
        
        Returns:
            bool: True if model is available, False otherwise
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            return self.model in result.stdout
        except Exception as e:
            logger.error(f"Failed to check model availability: {str(e)}")
            return False
            
    def _pull_model(self) -> bool:
        """
        Pull the required model if not available.
        
        Returns:
            bool: True if model was pulled successfully, False otherwise
        """
        try:
            logger.info(f"Pulling model {self.model}...")
            result = subprocess.run(
                ["ollama", "pull", self.model],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to pull model: {str(e)}")
            return False
            
    def initialize(self) -> bool:
        """
        Initialize the LLM environment.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if not self._check_ollama_installed():
            logger.error("Ollama is not installed. Please install it first.")
            return False
            
        if not self._check_model_available():
            if not self._pull_model():
                logger.error(f"Failed to pull model {self.model}")
                return False
                
        return True
        
    def generate_summary(self, prompt: str) -> Optional[str]:
        """
        Generate a summary using the local LLM.
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Returns:
            Optional[str]: The generated summary if successful, None otherwise
        """
        try:
            logger.info("Generating summary with local LLM...")
            
            # Create a temporary file for the prompt
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
                f.write(prompt)
                temp_file_path = f.name
            
            try:
                # Prepare the command
                cmd = [
                    "ollama", "run",
                    self.model,
                    f"$(cat {temp_file_path})"
                ]
                
                logger.debug(f"Running command: {' '.join(cmd)}")
                
                # Run the command with timeout
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    shell=True  # Enable shell to handle the $(cat) command
                )
                
                if result.returncode != 0:
                    logger.error(f"LLM generation failed: {result.stderr}")
                    return None
                    
                logger.debug(f"LLM output: {result.stdout[:200]}...")  # Log first 200 chars
                return result.stdout
                
            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)
                
        except subprocess.TimeoutExpired:
            logger.error(f"LLM generation timed out after {self.timeout} seconds")
            return None
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return None 