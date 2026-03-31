#!/bin/bash
set -e

# Setup script for Agentic RAG

echo "=== Setting up Agentic RAG ==="

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env from example if not exists
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# LangSmith
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=agentic-rag
EOF
    echo "Please edit .env with your API keys"
fi

echo "=== Setup complete! ==="
echo "Activate venv: source venv/bin/activate"
echo "Run API: uvicorn api.agentic_api:app --reload"
echo "Run CLI: python cli/agentic_chatbot.py"