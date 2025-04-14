# Transcription Summarizer

A Python-based system for summarizing HTML transcription content using local LLM inference.

## Features

- Fetches and parses HTML transcription content from SayIt archive
- Uses Ollama with Breeze/Qwen models for local inference
- Generates structured JSON summaries
- Dockerized for easy deployment

## Prerequisites

- Python 3.9+
- Docker (optional)
- Ollama runtime installed locally

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jeff14994/summary-LLM/blob/main/README.md
cd summarizer_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama (if not already installed):
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

4. Pull the required LLM model:
```bash
ollama pull jcai/breeze-7b-32k-instruct-v1_0:f16
```

## Usage

1. Run the summarizer:
```bash
python main.py --url "https://sayit.archive.tw/2025-02-02-bbc-採訪" --verbose
```

2. For Docker usage:
```bash
docker build -t summarizer .
docker run summarizer --url "https://sayit.archive.tw/2025-02-02-bbc-採訪"
```

## Project Structure

```
summarizer_project/
├── README.md
├── requirements.txt
├── Dockerfile
├── main.py
├── summarizer/
│   ├── __init__.py
│   ├── html_extractor.py
│   ├── prompt_builder.py
│   ├── local_llm.py
│   └── output_formatter.py
└── tests/
    └── test_cases.py
```

## Configuration

Create a `.env` file in the project root with the following variables:
```
OLLAMA_MODEL=jcai/breeze-7b-32k-instruct-v1_0
OUTPUT_DIR=./output
```

## License

MIT License 