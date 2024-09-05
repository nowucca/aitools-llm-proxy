import pytest
import openai
import anthropic
import os
import subprocess
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Dummy API keys and base URLs
OPENAI_API_KEY = "aitools-openai-api-key"
OPENAI_API_BASE_URL = "http://localhost:8000/openai/v1"
ANTHROPIC_API_KEY = "aitools-anthropic-api-key"
ANTHROPIC_API_BASE_URL = "http://localhost:8000/anthropic"


@pytest.fixture(scope="session", autouse=True)
def start_simpler_proxy():
    # Start the simpler_proxy process
    process = subprocess.Popen(
        ["uvicorn", "simpler_proxy:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    # Give the server a moment to start
    time.sleep(2)
    yield
    # Shutdown the simpler_proxy process
    process.terminate()
    process.wait()


@pytest.fixture(scope="function")
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
async def test_openai_api():
    # Initialize the OpenAI client with dummy values pointing to the proxy
    client = openai.OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE_URL
    )

    # Define the prompt and messages
    prompt = "Hello, how are you?"
    messages = [{"role": "user", "content": prompt}]

    # Make the request to the OpenAI API through the proxy
    response = client.chat.completions.create(
        model=os.getenv('OPENAI_API_MODEL'),
        messages=messages,
    ).choices[0].message.content

    # Check the response content
    assert response is not None
    print("OpenAI Response:", response)


@pytest.mark.asyncio
async def test_anthropic_api():
    # Initialize the Anthropic client with dummy values pointing to the proxy
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_API_BASE_URL)

    # Construct the request payload for Anthropic through the proxy
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, world"}
        ]
    )

    # Extract the relevant information from the response
    content_text = response.content[0].text  # Access the text of the first message

    # Check if the response content is as expected
    assert content_text is not None
    print("Anthropic Response:", content_text)
