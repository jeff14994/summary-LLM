# Transcription Summarizer

A Python-based system for summarizing HTML transcription content using local LLM inference.

[The Doc of implementation](https://hackmd.io/@jeff14994/rk3MDi3Cye)

## Features

- Fetches and parses HTML transcription content from SayIt archive
- Uses Ollama with Breeze/Qwen models for local inference
- Generates structured JSON summaries
- Dockerized for easy deployment
- **REST API server for backend integration**

## System Design of the ML System
```mermaid
flowchart TD
    A["Docker Container"]:::docker
    A --> B["User Interfaces"]:::interface

    subgraph "User Interfaces"
        B1["CLI Entry Points"]
        C["main.py"]:::cli
        C2["main1.py"]:::cli
        B1 --> C
        B1 --> C2
        B2["API Server (/summarize, /health, /docs, /redoc)"]:::api
    end

    %% Both CLI and API trigger the summarization pipeline
    B --> D["HTML Extractor"]:::module
    D --> E["Prompt Builder"]:::module
    E --> F["Local LLM"]:::module
    F --> G["Output Formatter"]:::module
    G --> H["JSON Summary Output"]:::module

    %% Configuration node
    O[".env Configuration"]:::config
    O --- F

    %% External dependency: Ollama Runtime
    R["Ollama Runtime"]:::external
    R --- F

    %% Testing Suite subgraph
    subgraph "Tests Suite"
        T1["Test HTML Extractor"]:::testing
        T2["Test Prompt Builder"]:::testing
        T3["Test Local LLM"]:::testing
        T4["Test Output Formatter"]:::testing
        T5["Test Parser"]:::testing
        T6["Test LLM"]:::testing
    end
    H --> I["Validation & Tests"]:::testing
    I --- T1
    I --- T2
    I --- T3
    I --- T4
    I --- T5
    I --- T6

    %% Click Events
    click D "https://github.com/jeff14994/summary-llm/blob/main/summarizer/html_extractor.py"
    click E "https://github.com/jeff14994/summary-llm/blob/main/summarizer/prompt_builder.py"
    click F "https://github.com/jeff14994/summary-llm/blob/main/summarizer/local_llm.py"
    click G "https://github.com/jeff14994/summary-llm/blob/main/summarizer/output_formatter.py"
    click C "https://github.com/jeff14994/summary-llm/blob/main/main.py"
    click C2 "https://github.com/jeff14994/summary-llm/blob/main/main1.py"
    click B2 "https://github.com/jeff14994/summary-llm/blob/main/api_server.py"
    click O "https://github.com/jeff14994/summary-llm/blob/main/.env"
    click T1 "https://github.com/jeff14994/summary-llm/blob/main/tests/test_html_extractor.py"
    click T2 "https://github.com/jeff14994/summary-llm/blob/main/tests/test_prompt_builder.py"
    click T3 "https://github.com/jeff14994/summary-llm/blob/main/tests/test_local_llm.py"
    click T4 "https://github.com/jeff14994/summary-llm/blob/main/tests/test_output_formatter.py"
    click T5 "https://github.com/jeff14994/summary-llm/blob/main/tests/test_parser.py"
    click T6 "https://github.com/jeff14994/summary-llm/blob/main/tests/test_llm.py"

    %% Style Definitions
    classDef docker fill:#D1C4E9,stroke:#000,stroke-width:2px;
    classDef interface fill:#FFDAB9,stroke:#000,stroke-width:2px;
    classDef cli fill:#CCE5FF,stroke:#000,stroke-width:2px;
    classDef api fill:#D1E7DD,stroke:#000,stroke-width:2px;
    classDef module fill:#FFF3CD,stroke:#000,stroke-width:2px;
    classDef config fill:#E2E3E5,stroke:#000,stroke-width:2px;
    classDef external fill:#B3E5FC,stroke:#000,stroke-width:2px;
    classDef testing fill:#F8D7DA,stroke:#000,stroke-width:2px;

```
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
ollama pull jcai/breeze-7b-32k-instruct-v1_0:q4_0
```

## Usage

### Command Line Interface

1. Run the summarizer:
```bash
python3 main.py --url "https://sayit.archive.tw/2025-02-02-bbc-採訪" --verbose
```

2. For Docker usage:
```bash
docker build -t summarizer .
docker run summarizer --url "https://sayit.archive.tw/2025-02-02-bbc-採訪"
```

### API Server

1. Start the API server:
```bash
python3 api_server.py
```

2. The server will be available at `http://localhost:8000`

3. Make API requests:
```bash
curl -X POST "http://localhost:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://sayit.archive.tw/2025-02-02-bbc-採訪", "verbose": true}'
```

4. API Documentation:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

5. Health Check:
```bash
curl "http://localhost:8000/health"
```

## Project Structure

```
summarizer_project/
├── README.md
├── requirements.txt
├── Dockerfile
├── main.py
├── api_server.py
├── summarizer/
│   ├── __init__.py
│   ├── html_extractor.py
│   ├── prompt_builder.py
│   ├── local_llm.py
│   └── output_formatter.py
└── tests/
    └── test_html_extractor.py   
    └── test_local_llm.py        
    └── test_parser.py
    └── test_llm.py              
    └── test_output_formatter.py 
    └── test_prompt_builder.py
```

## Configuration

Create a `.env` file in the project root with the following variables:
```
OLLAMA_MODEL=jcai/breeze-7b-32k-instruct-v1_0:q4_0
OUTPUT_DIR=./output
OLLAMA_TIMEOUT=30
OLLAMA_NUM_CTX=1024
OLLAMA_NUM_THREAD=8
MAX_CHUNK_SIZE=500
CHUNK_OVERLAP=100
MAX_WORKERS=4
```

## License

MIT License 
