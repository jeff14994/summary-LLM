"""
Test cases for the HTML extractor module.
"""

import pytest
from bs4 import BeautifulSoup
from summarizer.html_extractor import HTMLExtractor

# Sample HTML content for testing
SAMPLE_HTML = """
<div class="speech-wrapper">
    <div class="speech__meta-data">
        <span class="speech__meta-data__speaker-name">
            <a href="/speaker/唐鳳-3">唐鳳</a>
        </span>
    </div>
    <div class="speech__content">
        <p>嗯，你指的是 DeepSeek R1，是不是？就是最新的這個模型？</p>
    </div>
</div>
<div class="speech-wrapper">
    <div class="speech__meta-data">
        <span class="speech__meta-data__speaker-name">
            <a href="/speaker/主持人-1">主持人</a>
        </span>
    </div>
    <div class="speech__content">
        <p>是的，我們今天要討論的就是這個最新的AI模型。</p>
    </div>
</div>
"""

@pytest.fixture
def html_extractor():
    return HTMLExtractor()

def test_extract_speech_content(html_extractor):
    """Test extraction of content from a single speech wrapper."""
    soup = BeautifulSoup(SAMPLE_HTML, 'html.parser')
    speech_wrapper = soup.find('div', class_='speech-wrapper')
    
    result = html_extractor._extract_speech_content(speech_wrapper)
    assert result == "唐鳳: 嗯，你指的是 DeepSeek R1，是不是？就是最新的這個模型？"

def test_extract_transcription(html_extractor):
    """Test extraction of complete transcription."""
    result = html_extractor.extract_transcription(SAMPLE_HTML)
    expected = """唐鳳: 嗯，你指的是 DeepSeek R1，是不是？就是最新的這個模型？

主持人: 是的，我們今天要討論的就是這個最新的AI模型。"""
    assert result == expected

def test_extract_transcription_empty(html_extractor):
    """Test extraction with empty HTML."""
    result = html_extractor.extract_transcription("")
    assert result is None

def test_extract_transcription_no_speeches(html_extractor):
    """Test extraction with HTML containing no speeches."""
    html = "<div>Some other content</div>"
    result = html_extractor.extract_transcription(html)
    assert result is None

def test_extract_transcription_missing_content(html_extractor):
    """Test extraction with speech wrapper missing content."""
    html = """
    <div class="speech-wrapper">
        <div class="speech__meta-data">
            <span class="speech__meta-data__speaker-name">
                <a href="/speaker/唐鳳-3">唐鳳</a>
            </span>
        </div>
    </div>
    """
    result = html_extractor.extract_transcription(html)
    assert result is None

def test_extract_transcription_missing_speaker(html_extractor):
    """Test extraction with speech wrapper missing speaker."""
    html = """
    <div class="speech-wrapper">
        <div class="speech__content">
            <p>Some content without speaker</p>
        </div>
    </div>
    """
    result = html_extractor.extract_transcription(html)
    assert result == "Unknown Speaker: Some content without speaker" 