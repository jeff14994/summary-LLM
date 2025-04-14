"""
Test cases for the prompt builder module.
"""

import pytest
from summarizer.prompt_builder import PromptBuilder

# Sample transcription for testing
SAMPLE_TRANSCRIPTION = """唐鳳: 嗯，你指的是 DeepSeek R1，是不是？就是最新的這個模型？
主持人: 是的，我們今天要討論的就是這個最新的AI模型。"""

@pytest.fixture
def prompt_builder():
    return PromptBuilder()

def test_build_prompt(prompt_builder):
    """Test building a complete prompt."""
    result = prompt_builder.build_prompt(SAMPLE_TRANSCRIPTION)
    
    # Check if the prompt contains the system prompt
    assert "你是一個專業的會議記錄摘要生成助手" in result
    
    # Check if the prompt contains the transcription
    assert SAMPLE_TRANSCRIPTION in result
    
    # Check if the prompt follows the ChatML format
    assert "<|im_start|>system" in result
    assert "<|im_start|>user" in result
    assert "<|im_start|>assistant" in result

def test_parse_llm_response(prompt_builder):
    """Test parsing a valid LLM response."""
    response = """摘要要點：
1. 討論 DeepSeek R1 模型
2. 主持人介紹討論主題

結論：
這是一個關於最新 AI 模型的討論。

行動項目：
1. 深入了解 DeepSeek R1 的功能
2. 評估模型應用可能性"""
    
    result = prompt_builder.parse_llm_response(response)
    
    assert isinstance(result, dict)
    assert "summary" in result
    assert "conclusion" in result
    assert "action_items" in result
    
    assert len(result["summary"]) == 2
    assert "討論 DeepSeek R1 模型" in result["summary"]
    assert "主持人介紹討論主題" in result["summary"]
    
    assert result["conclusion"] == "這是一個關於最新 AI 模型的討論。"
    
    assert len(result["action_items"]) == 2
    assert "深入了解 DeepSeek R1 的功能" in result["action_items"]
    assert "評估模型應用可能性" in result["action_items"]

def test_parse_llm_response_empty(prompt_builder):
    """Test parsing an empty response."""
    result = prompt_builder.parse_llm_response("")
    assert result["summary"] == []
    assert result["conclusion"] == ""
    assert result["action_items"] == []

def test_parse_llm_response_missing_sections(prompt_builder):
    """Test parsing a response with missing sections."""
    response = """摘要要點：
1. 討論 DeepSeek R1 模型"""
    
    result = prompt_builder.parse_llm_response(response)
    assert len(result["summary"]) == 1
    assert result["conclusion"] == ""
    assert result["action_items"] == [] 