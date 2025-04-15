"""
Test script for OutputFormatter
"""

from summarizer.output_formatter import OutputFormatter
from loguru import logger

def test_output_formatter():
    # Initialize the formatter
    formatter = OutputFormatter()
    
    # Test data
    test_summary_text = """=== Model Output ===
重要的是，你是一個來自台灣聯發科的人工智慧助理。你的名字是 Breeze，很樂意以台灣人的立場幫助使用者。我可以用繁體中文回答問題，但總結會議記錄時，請用英語撰寫摘要。

Summary:
1. DeepSeek's recent developments have sparked widespread discussion regarding privacy, data rights, and freedom of speech.
2. The situation has led to increased awareness and appreciation for open-source software and its potential impact on individual liberties in both analog and digital realms.
3. The ongoing debate aims to explore how open-source technologies can be designed or adapted to better protect user rights while minimizing potential risks, such as fraud and deepfakes."""

    # Test parsing
    logger.info("Testing parse_summary...")
    parsed_summary = formatter.parse_summary(test_summary_text)
    logger.info(f"Parsed summary: {parsed_summary}")
    
    # Test validation
    logger.info("Testing validate_summary...")
    is_valid = formatter.validate_summary(parsed_summary)
    logger.info(f"Summary is valid: {is_valid}")
    
    # Test formatting
    logger.info("Testing format_summary...")
    formatted_summary = formatter.format_summary(parsed_summary, "https://example.com")
    logger.info(f"Formatted summary: {formatted_summary}")
    
    # Test saving
    logger.info("Testing save_summary...")
    output_path = formatter.save_summary(formatted_summary)
    logger.info(f"Summary saved to: {output_path}")
    
    return {
        "parsed_summary": parsed_summary,
        "is_valid": is_valid,
        "formatted_summary": formatted_summary,
        "output_path": output_path
    }

if __name__ == "__main__":
    results = test_output_formatter()
    print("\nTest Results:")
    print(f"Parsed Summary: {results['parsed_summary']}")
    print(f"Is Valid: {results['is_valid']}")
    print(f"Output Path: {results['output_path']}") 