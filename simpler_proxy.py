import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request

import anthropic_proxy

# Load .env file
load_dotenv()
import openai_proxy

VERBOSE_LOGGING = os.getenv('VERBOSE_LOGGING', 'false').lower() == 'true'

logger = logging.getLogger(__name__)

# Configure logging based on VERBOSE_LOGGING
if VERBOSE_LOGGING:
    # Configure logging with a specific format including a timestamp
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("httpcore").setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

app = FastAPI()


@app.api_route("/openai/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_openai(path: str, request: Request):
    return await openai_proxy.proxy_openai(path, request)


@app.api_route("/anthropic/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def route_anthropic(path: str, request: Request):
    return await anthropic_proxy.proxy_anthropic(path, request)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
