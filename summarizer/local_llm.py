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
import time
import sys

class LocalLLM:
    def __init__(self):
        load_dotenv()
        # Use the full model name including quantization suffix
        self.model = os.getenv("OLLAMA_MODEL", "jcai/breeze-7b-32k-instruct-v1_0:q4_0")
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))
        # Add parameters for faster inference
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "2048"))  # Context window size
        self.num_thread = int(os.getenv("OLLAMA_NUM_THREAD", "4"))  # Number of threads
        self.num_gpu = int(os.getenv("OLLAMA_NUM_GPU", "1"))  # Number of GPU layers
        
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
        
    def _print_progress(self, start_time: float, timeout: int):
        """Print a progress indicator for the inference process."""
        elapsed = time.time() - start_time
        progress = min(100, int((elapsed / timeout) * 100))
        sys.stdout.write(f"\rInference progress: [{progress}%] {'=' * (progress//2)}{' ' * (50-progress//2)}")
        sys.stdout.flush()
        
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
            start_time = time.time()
            
            # Create a temporary file for the prompt
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
                f.write(prompt)
                temp_file_path = f.name
            
            try:
                # Prepare the command with optimization parameters
                cmd = (
                    f"OLLAMA_NUM_CTX=1024 "
                    f"OLLAMA_NUM_THREAD=8 "
                    f"OLLAMA_NUM_GPU=1 "
                    f"cat {temp_file_path} | "
                    f"ollama run {self.model}"
                )
                
                logger.debug(f"Running command: {cmd}")
                
                # Start the process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=True,
                    env=dict(os.environ, OLLAMA_NUM_CTX="1024", OLLAMA_NUM_THREAD="8", OLLAMA_NUM_GPU="1")
                )
                
                # Monitor the process and show progress
                output = []
                while True:
                    # Check if process is still running
                    if process.poll() is not None:
                        break
                    
                    # Print progress
                    self._print_progress(start_time, self.timeout)
                    
                    # Check for timeout
                    if time.time() - start_time > self.timeout:
                        process.terminate()
                        logger.error(f"LLM generation timed out after {self.timeout} seconds")
                        return None
                    
                    # Read output if available
                    line = process.stdout.readline()
                    if line:
                        output.append(line)
                        logger.debug(f"Model output: {line.strip()}")
                    
                    time.sleep(0.1)  # Small delay to prevent high CPU usage
                
                # Get final output and error
                stdout, stderr = process.communicate()
                output.extend(stdout.splitlines())
                
                if process.returncode != 0:
                    logger.error(f"LLM generation failed: {stderr}")
                    return None
                
                final_output = ''.join(output)
                if not final_output.strip():
                    logger.error("LLM returned empty output")
                    return None
                
                print()  # New line after progress bar
                logger.info("Inference completed successfully")
                logger.debug(f"Final output: {final_output[:200]}...")  # Log first 200 chars
                return final_output
                
            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return None 