from summarizer.local_llm import LocalLLM
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, level="DEBUG")

def test_llm():
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
    summary = llm.generate_summary(test_prompt)
    
    if summary:
        logger.info("Test successful! Here's the summary:")
        logger.info(summary)
    else:
        logger.error("Test failed - no summary generated")

if __name__ == "__main__":
    test_llm() 