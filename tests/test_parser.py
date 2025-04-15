from summarizer.prompt_builder import PromptBuilder
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, level="DEBUG")

def test_parser():
    # Initialize the prompt builder
    prompt_builder = PromptBuilder()
    
    # Test raw summary
    raw_summary = """=== Model Output ===
Breeze 是個台湾联发科 AI 助手。你最想做 Taiwanese 立場，用正体中文回答問嘅。
1. 摘要要點：列表 3-5 個最重要嘅討嚟點
2. 結論：簡要總结會議嘅主要結論
3.

1. DeepSeek's recent events have sparked widespread discussion about privacy, rights, and open-source development.
2. The conversation should focus on how open-source methods can balance the promotion of human rights while minimizing potential risks like fraud and deepfakes.
3. The assistant discusses different AI models like Perplexity R1, O3 Mini High, and Gemini Thinking as alternatives to DeepSeek.
4. It is suggested that focusing on common anti-malicious tools or developing new open-source methods may be a way forward in balancing the benefits and risks of open-source development."""

    # Parse the summary
    logger.info("Testing parser with raw summary...")
    parsed_summary = prompt_builder.parse_llm_response(raw_summary)
    
    # Print the results
    logger.info("\nParsed Summary:")
    logger.info("=" * 80)
    logger.info(f"Summary points: {parsed_summary['summary']}")
    logger.info(f"Conclusion: {parsed_summary['conclusion']}")
    logger.info(f"Action items: {parsed_summary['action_items']}")
    logger.info("=" * 80)

if __name__ == "__main__":
    test_parser() 