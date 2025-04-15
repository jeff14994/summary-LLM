"""
Prompt Builder Module

This module handles constructing effective prompts for the LLM to generate summaries.
"""

from typing import List, Dict, Any
from loguru import logger

class PromptBuilder:
    def __init__(self):
        self.system_prompt = """你是一個專業的會議記錄摘要生成助手。你的任務是根據提供的會議記錄內容，生成結構化的摘要。
請遵循以下格式生成摘要：
1. 摘要要點：列出3-5個最重要的討論點
2. 結論：簡要總結會議的主要結論
3. 行動項目：列出需要跟進的具體行動項目

請確保摘要：
- 保持客觀中立
- 突出關鍵決策和行動項目
- 使用清晰簡潔的語言
- 保持原始內容的準確性"""

    def build_prompt(self, transcription: str) -> str:
        """
        Build a complete prompt for the LLM including system instructions and transcription content.
        
        Args:
            transcription (str): The transcription text to summarize
            
        Returns:
            str: The complete prompt for the LLM
        """
        try:
            prompt = f"""<|im_start|>system
{self.system_prompt}
<|im_end|>

<|im_start|>user
請根據以下會議記錄生成摘要：

{transcription}
<|im_end|>

<|im_start|>assistant
好的，我將根據提供的會議記錄生成結構化摘要：

"""
            return prompt
        except Exception as e:
            logger.error(f"Failed to build prompt: {str(e)}")
            raise

    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM's response into a structured format.
        
        Args:
            response (str): The raw response from the LLM
            
        Returns:
            Dict[str, Any]: Structured summary containing key points, conclusion, and action items
        """
        try:
            # Initialize the summary structure
            summary = {
                "summary": [],
                "conclusion": "",
                "action_items": []
            }
            
            # Split the response into lines and clean them
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for section headers in both English and Chinese
                if any(header in line.lower() for header in ["summary:", "摘要要點", "要點"]):
                    current_section = "summary"
                    continue
                elif any(header in line.lower() for header in ["conclusion:", "結論"]):
                    current_section = "conclusion"
                    continue
                elif any(header in line.lower() for header in ["action items:", "行動項目", "行動"]):
                    current_section = "action_items"
                    continue
                
                # Skip lines that are clearly not content
                if line.startswith(("=== Model", "You are", "Please", "IMPORTANT")):
                    continue
                
                # Add content to appropriate section
                if current_section == "summary":
                    if line.startswith(("*", "-", "•", "1.", "2.", "3.")):
                        summary["summary"].append(line.lstrip("*•-123. "))
                elif current_section == "conclusion":
                    if not summary["conclusion"]:  # Only take the first conclusion line
                        summary["conclusion"] = line
                elif current_section == "action_items":
                    if line.startswith(("*", "-", "•", "1.", "2.", "3.")):
                        summary["action_items"].append(line.lstrip("*•-123. "))
            
            # If no conclusion was found, use the last summary point
            if not summary["conclusion"] and summary["summary"]:
                summary["conclusion"] = summary["summary"][-1]
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            raise
