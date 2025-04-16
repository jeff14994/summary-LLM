from summarizer.local_llm import LocalLLM
from loguru import logger
import sys
import time
from datetime import datetime
import os

# Configure logger
logger.remove()
logger.add(sys.stderr, level="DEBUG")

def print_progress(start_time: float, timeout: int, output_lines: int = 0, status: str = "Running"):
    """Print a progress indicator for the inference process."""
    elapsed = time.time() - start_time
    progress = min(100, int((elapsed / timeout) * 100))
    elapsed_str = datetime.fromtimestamp(elapsed).strftime('%M:%S')
    status_line = f"Status: {status} | Time: {elapsed_str} | Output: {output_lines} lines | Progress: [{progress}%] {'=' * (progress//2)}{' ' * (50-progress//2)}"
    sys.stdout.write(f"\r{status_line}")
    sys.stdout.flush()

def test_llm():
    # Create output directory if it doesn't exist
    output_dir = "llm_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"llm_output_{timestamp}.txt")
    
    # Initialize the LLM
    llm = LocalLLM()
    
    # Test initialization
    if not llm.initialize():
        logger.error("Failed to initialize LLM")
        return
    
    # Test prompt
    test_prompt = """You are a helpful AI assistant. Please provide a brief summary of the following text:

The quick brown fox jumps over the lazy dog. This is a classic pangram that contains every letter of the English alphabet at least once. It's often used for typing practice and testing fonts.

Please keep your response concise and to the point."""

    # Generate summary
    logger.info("Testing LLM with a simple prompt...")
    logger.info(f"Model: {llm.model}")
    logger.info(f"Timeout: {llm.timeout} seconds")
    logger.info(f"Context window: {llm.num_ctx}")
    logger.info(f"Threads: {llm.num_thread}")
    logger.info(f"GPU layers: {llm.num_gpu}")
    logger.info(f"Output will be saved to: {output_file}")
    
    start_time = time.time()
    last_update = time.time()
    update_interval = 0.5  # Update progress every 0.5 seconds
    output_lines = 0
    
    def progress_callback(line: str):
        nonlocal output_lines, last_update
        output_lines += 1
        current_time = time.time()
        if current_time - last_update >= update_interval:
            print_progress(start_time, llm.timeout, output_lines)
            last_update = current_time
        logger.debug(f"Model output: {line.strip()}")
    
    # Generate summary with progress monitoring
    summary = llm.generate_summary(test_prompt)
    
    print()  # New line after progress bar
    
    if summary:
        # Write output to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== Model Configuration ===\n")
            f.write(f"Model: {llm.model}\n")
            f.write(f"Timeout: {llm.timeout} seconds\n")
            f.write(f"Context window: {llm.num_ctx}\n")
            f.write(f"Threads: {llm.num_thread}\n")
            f.write(f"GPU layers: {llm.num_gpu}\n\n")
            
            f.write("=== Test Prompt ===\n")
            f.write(test_prompt)
            f.write("\n\n")
            
            f.write("=== Model Output ===\n")
            f.write(summary)
        
        logger.info("Test successful! Summary saved to file.")
        logger.info(f"Output file: {output_file}")
        
        # Also print summary to console
        logger.info("Here's the summary:")
        logger.info("=" * 80)
        logger.info(summary)
        logger.info("=" * 80)
    else:
        logger.error("Test failed - no summary generated")

if __name__ == "__main__":
    test_llm() 