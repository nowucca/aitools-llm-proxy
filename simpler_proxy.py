import traceback

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import StreamingResponse
from httpx import Timeout, NetworkError, HTTPStatusError, TooManyRedirects, InvalidURL, ConnectTimeout, ReadTimeout, \
    RequestError, PoolTimeout
from starlette.background import BackgroundTask
import httpx
import logging
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE_URL = os.getenv('OPENAI_API_BASE_URL')
OPENAI_ORG = os.getenv('OPENAI_ORG')
VERBOSE_LOGGING = os.getenv('VERBOSE_LOGGING', 'false').lower() == 'true'
# Load and validate OPENAI_TIMEOUT
OPENAI_TIMEOUT = int(os.getenv('OPENAI_TIMEOUT', 60))
OPENAI_TIMEOUT = max(1, min(OPENAI_TIMEOUT, 120))  # Ensure between 1 and 120 seconds

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
client = httpx.AsyncClient(base_url=OPENAI_API_BASE_URL)

async def clean_headers(headers: dict, api_key: str, org: str) -> dict:
    cleaned_headers = {k: v for k, v in headers.items() if k.lower() not in ['host', 'authorization']}
    cleaned_headers['Authorization'] = f'Bearer {api_key}'
    cleaned_headers['OpenAI-Organization'] = org
    return cleaned_headers


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_openai(path: str, request: Request):
    """
        Proxy requests to the OpenAI API.

        :param path: The full path from the incoming request, including the 'openai/' prefix if present.
        :param request: The incoming request with headers, method, etc.
        """

    # Remove 'openai/' prefix from the path, if it exists
    api_path = path[len('openai/'):] if path.startswith('openai/') else path

    request_method = request.method
    request_headers = dict(request.headers)
    request_content = request.stream()

    cleaned_headers = await clean_headers(request_headers, OPENAI_API_KEY, OPENAI_ORG)

    try:
        url = httpx.URL(path=api_path, query=request.url.query.encode("utf-8"))
        rp_req = client.build_request(request_method, url, timeout=OPENAI_TIMEOUT, headers=cleaned_headers,
                                      content=request_content)
        rp_resp = await client.send(rp_req, stream=True)

    except NetworkError as e:
        traceback.print_exc()
        raise HTTPException(status_code=503, detail="Service Unavailable [aitools]")
    except TooManyRedirects as e:
        traceback.print_exc()
        raise HTTPException(status_code=310, detail="Too Many Redirects [aitools]")
    except InvalidURL as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="Invalid URL [aitools]")
    except ConnectTimeout as e:
        traceback.print_exc()
        raise HTTPException(status_code=408, detail="Connect Timeout [aitools]")
    except ReadTimeout as e:
        traceback.print_exc()
        raise HTTPException(status_code=408, detail="Read Timeout [aitools]")
    except PoolTimeout as e:
        raise HTTPException(status_code=503, detail="Service Unavailable due to Pool Timeout [aitools]")
    except RequestError as e:
        raise HTTPException(status_code=500, detail="Request Error [aitools]")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e) + " [aitools]")

    if rp_resp.status_code != 200:
        raise HTTPException(status_code=rp_resp.status_code, detail=rp_resp.reason_phrase)

    return StreamingResponse(
        rp_resp.aiter_raw(),
        status_code=rp_resp.status_code,
        headers=rp_resp.headers,
        background=BackgroundTask(rp_resp.aclose)
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
