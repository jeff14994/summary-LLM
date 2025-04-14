"""
Test cases for the output formatter module.
"""

import pytest
import json
import os
from datetime import datetime
from summarizer.output_formatter import OutputFormatter

# Sample summary for testing
SAMPLE_SUMMARY = {
    "summary": [
        "討論 DeepSeek R1 模型",
        "主持人介紹討論主題"
    ],
    "conclusion": "這是一個關於最新 AI 模型的討論。",
    "action_items": [
        "深入了解 DeepSeek R1 的功能",
        "評估模型應用可能性"
    ]
}

@pytest.fixture
def output_formatter(tmp_path):
    """Create an output formatter with a temporary directory."""
    os.environ["OUTPUT_DIR"] = str(tmp_path)
    return OutputFormatter()

def test_format_summary(output_formatter):
    """Test formatting a summary with metadata."""
    url = "https://example.com/transcription"
    result = output_formatter.format_summary(SAMPLE_SUMMARY, url)
    
    assert "metadata" in result
    assert "content" in result
    
    metadata = result["metadata"]
    assert metadata["source_url"] == url
    assert "generated_at" in metadata
    assert metadata["version"] == "1.0"
    
    content = result["content"]
    assert content == SAMPLE_SUMMARY

def test_save_summary(output_formatter):
    """Test saving a formatted summary to a file."""
    url = "https://example.com/transcription"
    formatted_summary = output_formatter.format_summary(SAMPLE_SUMMARY, url)
    
    filepath = output_formatter.save_summary(formatted_summary)
    
    assert os.path.exists(filepath)
    assert filepath.startswith(str(output_formatter.output_dir))
    assert filepath.endswith(".json")
    
    # Verify file content
    with open(filepath, 'r', encoding='utf-8') as f:
        saved_content = json.load(f)
        assert saved_content == formatted_summary

def test_validate_summary(output_formatter):
    """Test validating a valid summary."""
    assert output_formatter.validate_summary(SAMPLE_SUMMARY) is True

def test_validate_summary_missing_fields(output_formatter):
    """Test validating a summary with missing fields."""
    invalid_summary = {
        "summary": [],
        "conclusion": ""
        # Missing action_items
    }
    assert output_formatter.validate_summary(invalid_summary) is False

def test_validate_summary_wrong_types(output_formatter):
    """Test validating a summary with wrong field types."""
    invalid_summary = {
        "summary": "Not a list",  # Should be a list
        "conclusion": ["Not a string"],  # Should be a string
        "action_items": "Not a list"  # Should be a list
    }
    assert output_formatter.validate_summary(invalid_summary) is False 