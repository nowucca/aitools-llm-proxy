import pytest
from dotenv import load_dotenv
from httpx import AsyncClient
from simpler_proxy import app

# Load environment variables from .env file
load_dotenv()

@pytest.fixture(scope="function")
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio()
async def test_openai_through_proxy():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Construct the request payload for OpenAI
        payload = {
            "model": "gpt-4-turbo-preview",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 15
        }
        # Make a POST request to the FastAPI application, which proxies to OpenAI
        response = await client.post("/openai/v1/chat/completions", json=payload)

        # Check the response status code
        assert response.status_code==200
        # Optionally, print the response text for inspection
        print(response.json())


@pytest.mark.asyncio
async def test_anthropic_through_proxy():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Construct the request payload for Anthropic
        payload = {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello, world"}]
        }
        # Make a POST request to the FastAPI application, which proxies to Anthropic
        response = await client.post("/anthropic/v1/messages", json=payload)

        # Check the response status code
        assert response.status_code==200
        # Optionally, print the response text for inspection
        print(response.json())
