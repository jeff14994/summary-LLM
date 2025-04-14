#!/usr/bin/env python3
"""
Transcription Summarizer

A tool for summarizing HTML transcription content using local LLM inference.
"""

import argparse
import sys
from loguru import logger
from summarizer.html_extractor import HTMLExtractor
from summarizer.prompt_builder import PromptBuilder
from summarizer.local_llm import LocalLLM
from summarizer.output_formatter import OutputFormatter

def setup_logging():
    """Configure logging settings."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Summarize transcription content from a URL")
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL of the transcription to summarize"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()

def main():
    """Main entry point of the application."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging()
    if args.verbose:
        logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])
    
    try:
        # Initialize components
        html_extractor = HTMLExtractor()
        prompt_builder = PromptBuilder()
        local_llm = LocalLLM()
        output_formatter = OutputFormatter()
        
        # Initialize LLM
        logger.info("Initializing LLM...")
        if not local_llm.initialize():
            logger.error("Failed to initialize LLM")
            sys.exit(1)
        
        # Extract transcription
        logger.info(f"Fetching transcription from {args.url}...")
        transcription = html_extractor.process_url(args.url)
        if not transcription:
            logger.error("Failed to extract transcription")
            sys.exit(1)
        
        # Build prompt
        logger.info("Building prompt...")
        prompt = prompt_builder.build_prompt(transcription)
        
        # Generate summary
        logger.info("Generating summary...")
        raw_summary = local_llm.generate_summary(prompt)
        if not raw_summary:
            logger.error("Failed to generate summary")
            sys.exit(1)
        
        # Parse and validate summary
        logger.info("Parsing summary...")
        summary = prompt_builder.parse_llm_response(raw_summary)
        if not output_formatter.validate_summary(summary):
            logger.error("Invalid summary format")
            sys.exit(1)
        
        # Format and save summary
        logger.info("Saving summary...")
        formatted_summary = output_formatter.format_summary(summary, args.url)
        output_path = output_formatter.save_summary(formatted_summary)
        
        logger.success(f"Summary successfully generated and saved to {output_path}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 