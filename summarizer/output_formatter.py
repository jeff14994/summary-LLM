"""
Output Formatter Module

This module handles formatting and saving the summary output.
"""

import json
import os
from typing import Dict, Any
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

class OutputFormatter:
    def __init__(self):
        load_dotenv()
        self.output_dir = os.getenv("OUTPUT_DIR", "./output")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def format_summary(self, summary: Dict[str, Any], url: str) -> Dict[str, Any]:
        """
        Format the summary with metadata.
        
        Args:
            summary (Dict[str, Any]): The summary content
            url (str): The source URL
            
        Returns:
            Dict[str, Any]: Formatted summary with metadata
        """
        return {
            "metadata": {
                "source_url": url,
                "generated_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "content": summary
        }
        
    def save_summary(self, formatted_summary: Dict[str, Any]) -> str:
        """
        Save the formatted summary to a JSON file.
        
        Args:
            formatted_summary (Dict[str, Any]): The formatted summary
            
        Returns:
            str: Path to the saved file
        """
        try:
            # Generate filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summary_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(formatted_summary, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Summary saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save summary: {str(e)}")
            raise
            
    def validate_summary(self, summary: Dict[str, Any]) -> bool:
        """
        Validate the summary structure.
        
        Args:
            summary (Dict[str, Any]): The summary to validate
            
        Returns:
            bool: True if summary is valid, False otherwise
        """
        required_fields = ["summary", "conclusion", "action_items"]
        
        # Check if all required fields are present
        if not all(field in summary for field in required_fields):
            logger.error("Summary missing required fields")
            return False
            
        # Check if fields have correct types
        if not isinstance(summary["summary"], list):
            logger.error("Summary points must be a list")
            return False
        if not isinstance(summary["conclusion"], str):
            logger.error("Conclusion must be a string")
            return False
        if not isinstance(summary["action_items"], list):
            logger.error("Action items must be a list")
            return False
            
        return True 