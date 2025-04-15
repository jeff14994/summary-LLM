"""
API Server for Summarization Service

This module provides a FastAPI server to serve the summarization functionality.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional
import uvicorn
from loguru import logger
from summarizer.prompt_builder import PromptBuilder
from summarizer.local_llm import LocalLLM
from summarizer.output_formatter import OutputFormatter
from summarizer.html_extractor import HTMLExtractor
import requests
from bs4 import BeautifulSoup
import sys

app = FastAPI(
    title="Summarization API",
    description="API for generating summaries from URL content",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
html_extractor = HTMLExtractor()
prompt_builder = PromptBuilder()
llm = LocalLLM()
output_formatter = OutputFormatter()

class SummarizeRequest(BaseModel):
    url: HttpUrl
    verbose: Optional[bool] = False

class SummarizeResponse(BaseModel):
    summary: dict
    success: bool
    message: Optional[str] = None
    output_path: Optional[str] = None
    logs: Optional[list] = None

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Generate a summary from the content of the provided URL.
    
    Args:
        request (SummarizeRequest): The request containing URL to summarize and verbose flag
        
    Returns:
        SummarizeResponse: The generated summary and status
    """
    try:
        # Configure logging level based on verbose flag
        if request.verbose:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.remove()
            logger.add(sys.stderr, level="INFO")
        
        logger.info(f"Received summarization request for URL: {request.url}")
        logs = []
        
        # Initialize LLM
        logger.info("Initializing LLM...")
        if not llm.initialize():
            raise HTTPException(status_code=500, detail="Failed to initialize LLM")
        logs.append("LLM initialized successfully")
        
        # Extract transcription
        logger.info(f"Fetching transcription from {request.url}...")
        transcription = html_extractor.process_url(str(request.url))
        if not transcription:
            raise HTTPException(status_code=400, detail="Failed to extract transcription")
        logs.append("Transcription extracted successfully")
        
        # Build prompt
        logger.info("Building prompt...")
        prompt = prompt_builder.build_prompt(transcription)
        logs.append("Prompt built successfully")
        
        # Generate summary
        logger.info("Generating summary...")
        raw_summary = llm.generate_summary(prompt)
        if not raw_summary:
            raise HTTPException(status_code=500, detail="Failed to generate summary")
        logs.append("Summary generated successfully")
        
        # Parse and validate summary
        logger.info("Parsing summary...")
        summary = prompt_builder.parse_llm_response(raw_summary)
        if not output_formatter.validate_summary(summary):
            logger.error(f"Invalid summary format: {summary}")
            raise HTTPException(status_code=500, detail="Invalid summary format")
        logs.append("Summary parsed and validated successfully")
        
        # Format and save summary
        logger.info("Saving summary...")
        formatted_summary = output_formatter.format_summary(summary, str(request.url))
        output_path = output_formatter.save_summary(formatted_summary)
        logs.append(f"Summary saved to {output_path}")
        
        # Log the final summary for debugging
        logger.debug(f"Final summary: {formatted_summary}")
        
        return SummarizeResponse(
            summary=formatted_summary,
            success=True,
            message="Summary generated successfully",
            output_path=output_path,
            logs=logs if request.verbose else None
        )
        
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Health status
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True) 