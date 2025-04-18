"""
Local LLM Module

This module handles interaction with the Ollama runtime for local LLM inference.
"""

import subprocess
import json
from typing import Optional, Dict, Any, List
from loguru import logger
import os
from dotenv import load_dotenv
import time
import sys
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

class LocalLLM:
    def __init__(self):
        load_dotenv()
        # Use the full model name including quantization suffix
        self.model = os.getenv("OLLAMA_MODEL", "jcai/breeze-7b-32k-instruct-v1_0:q4_0")
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "0"))  # 0 means no timeout
        # Add parameters for faster inference
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "1024"))  # Reduced context window
        self.num_thread = int(os.getenv("OLLAMA_NUM_THREAD", "8"))  # Increased threads
        self.num_gpu = int(os.getenv("OLLAMA_NUM_GPU", "1"))  # Number of GPU layers
        # Create output directory
        self.output_dir = "llm_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        # Process tracking
        self.current_process = None
        self.is_running = False
        # Chunking parameters
        self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "500"))  # Reduced chunk size
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))  # Reduced overlap
        # Parallel processing
        self.max_workers = int(os.getenv("MAX_WORKERS", "4"))  # Number of parallel workers
        
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
        
    def _print_progress(self, start_time: float, timeout: int, output_lines: int = 0, status: str = "Running"):
        """Print a progress indicator for the inference process."""
        elapsed = time.time() - start_time
        if timeout > 0:
            progress = min(100, int((elapsed / timeout) * 100))
            progress_str = f"[{progress}%] {'=' * (progress//2)}{' ' * (50-progress//2)}"
        else:
            progress_str = "[No timeout]"
        elapsed_str = datetime.fromtimestamp(elapsed).strftime('%M:%S')
        status_line = f"Status: {status} | Time: {elapsed_str} | Output: {output_lines} lines | Progress: {progress_str}"
        sys.stdout.write(f"\r{status_line}")
        sys.stdout.flush()
        
    def _save_output(self, prompt: str, summary: str, output_file: str):
        """Save the model output to a file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== Model Configuration ===\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Timeout: {self.timeout} seconds\n")
            f.write(f"Context window: {self.num_ctx}\n")
            f.write(f"Threads: {self.num_thread}\n")
            f.write(f"GPU layers: {self.num_gpu}\n\n")
            
            f.write("=== Test Prompt ===\n")
            f.write(prompt)
            f.write("\n\n")
            
            f.write("=== Model Output ===\n")
            f.write(summary)
            
    def cancel_generation(self):
        """Cancel the current generation process if it's running."""
        if self.is_running and self.current_process:
            logger.info("Cancelling current generation...")
            self.current_process.terminate()
            self.is_running = False
            self.current_process = None
            logger.info("Generation cancelled")
        else:
            logger.info("No generation process is currently running")
        
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text (str): The text to split
            
        Returns:
            List[str]: List of text chunks
        """
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed the chunk size, start a new chunk
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep some sentences for overlap
                overlap_size = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
            
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
        
    def _process_chunk_parallel(self, chunk_info: tuple) -> tuple:
        """
        Process a single chunk of text in parallel.
        
        Args:
            chunk_info: Tuple of (chunk_index, chunk, is_first, is_last)
            
        Returns:
            tuple: (chunk_index, summary)
        """
        chunk_index, chunk, is_first, is_last = chunk_info
        prompt = f"""You are a helpful AI assistant. Please process the following text{' (this is the first part)' if is_first else (' (this is the last part)' if is_last else '')}:

{chunk}

Please provide a concise summary of this section. Focus on the key points and main ideas.
IMPORTANT: 
1. Use only English for the summary
2. Format the output as a clean, numbered list
3. Each point should be a complete sentence
4. Avoid any special characters or formatting markers
5. Keep the summary focused and concise"""

        summary = self._generate_with_prompt(prompt)
        return (chunk_index, summary)
        
    def _generate_with_prompt(self, prompt: str) -> Optional[str]:
        """
        Generate text using the given prompt.
        
        Args:
            prompt (str): The prompt to use
            
        Returns:
            Optional[str]: The generated text if successful, None otherwise
        """
        try:
            # Create a temporary file for the prompt
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
                f.write(prompt)
                temp_file_path = f.name
            
            try:
                # Prepare the command with optimization parameters
                cmd = (
                    f"OLLAMA_NUM_CTX={self.num_ctx} "
                    f"OLLAMA_NUM_THREAD={self.num_thread} "
                    f"OLLAMA_NUM_GPU={self.num_gpu} "
                    f"cat {temp_file_path} | "
                    f"ollama run {self.model}"
                )
                
                # Start the process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=True,
                    env=dict(os.environ, 
                            OLLAMA_NUM_CTX=str(self.num_ctx),
                            OLLAMA_NUM_THREAD=str(self.num_thread),
                            OLLAMA_NUM_GPU=str(self.num_gpu))
                )
                
                # Get the output
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"Chunk processing failed: {stderr}")
                    return None
                
                return stdout.strip()
                
            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Failed to process chunk: {str(e)}")
            return None
        
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
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"llm_output_{timestamp}.txt")
            logger.info(f"Output will be saved to: {output_file}")
            
            # Split the prompt into chunks
            chunks = self._split_into_chunks(prompt)
            logger.info(f"Split text into {len(chunks)} chunks")
            
            # Prepare chunk information for parallel processing
            chunk_info = [
                (i, chunk, i == 0, i == len(chunks) - 1)
                for i, chunk in enumerate(chunks)
            ]
            
            # Process chunks in parallel
            summaries = [None] * len(chunks)
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_chunk = {
                    executor.submit(self._process_chunk_parallel, info): info[0]
                    for info in chunk_info
                }
                
                for future in as_completed(future_to_chunk):
                    chunk_index, summary = future.result()
                    if summary:
                        # Clean up the summary
                        summary = summary.strip()
                        # Remove any special characters or formatting markers
                        summary = re.sub(r'<\|.*?\|>', '', summary)
                        summaries[chunk_index] = summary
                        logger.info(f"Completed chunk {chunk_index + 1}/{len(chunks)}")
                    else:
                        logger.error(f"Failed to process chunk {chunk_index + 1}")
                        return None
            
            # Combine summaries and clean up
            final_summary = "\n\n".join(filter(None, summaries))
            final_summary = re.sub(r'\n{3,}', '\n\n', final_summary)  # Remove excessive newlines
            
            # Save output to file
            self._save_output(prompt, final_summary, output_file)
            logger.info(f"Output saved to: {output_file}")
            
            return final_summary
                
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return None 