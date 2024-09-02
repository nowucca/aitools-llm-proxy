# openai_proxy.py
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import Request, HTTPException, FastAPI
from fastapi.responses import StreamingResponse
from httpx import ReadTimeout, ConnectTimeout, PoolTimeout, NetworkError, TooManyRedirects, InvalidURL, RequestError
from starlette.background import BackgroundTask
import logging
import os
import traceback
from client_manager import ClientManager

load_dotenv()


# Read environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE_URL = os.getenv('OPENAI_API_BASE_URL')
OPENAI_ORG = os.getenv('OPENAI_ORG')
# Load and validate OPENAI_TIMEOUT
OPENAI_TIMEOUT = int(os.getenv('OPENAI_TIMEOUT', 60))
OPENAI_TIMEOUT = max(1, min(OPENAI_TIMEOUT, 120))  # Ensure between 1 and 120 seconds

logger = logging.getLogger(__name__)
client_manager = ClientManager(base_url=OPENAI_API_BASE_URL, timeout=OPENAI_TIMEOUT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client_manager
    yield
    # Clean up the ML models and release the resources
    await client_manager.close()


async def clean_headers(headers: dict, api_key: str, org: str) -> dict:
    cleaned_headers = {k: v for k, v in headers.items() if k.lower() not in ['host', 'authorization']}
    cleaned_headers['Authorization'] = f'Bearer {api_key}'
    cleaned_headers['OpenAI-Organization'] = org
    return cleaned_headers


async def proxy_openai(path: str, request: Request):
    client = await client_manager.get_client()
    api_path = path[len('openai/'):] if path.startswith('openai/') else path

    request_method = request.method
    request_headers = dict(request.headers)
    request_content = request.stream()

    cleaned_headers = await clean_headers(request_headers, OPENAI_API_KEY, OPENAI_ORG)

    try:
        url = httpx.URL(path=api_path, query=request.url.query.encode("utf-8"))
        rp_req = client.build_request(request_method, url, timeout=60, headers=cleaned_headers, content=request_content)
        rp_resp = await client.send(rp_req, stream=True)
    except ReadTimeout as e:
        error_detail = f"Read Timeout [aitools] - Error: {e}"
        logger.error(f"ReadTimeout encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=408, detail=error_detail)
    except ConnectTimeout as e:
        await client_manager.increment_error()
        error_detail = f"Connect Timeout [aitools] - Error: {e}"
        logger.error(f"ConnectTimeout encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=408, detail=error_detail)
    except PoolTimeout as e:
        await client_manager.increment_error()
        error_detail = f"Service Unavailable due to Pool Timeout [aitools] - Error: {e}"
        logger.error(f"PoolTimeout encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=503, detail=error_detail)
    except NetworkError as e:
        await client_manager.increment_error()
        error_detail = f"Service Unavailable [aitools] - Error: {e}"
        logger.error(f"NetworkError encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=503, detail=error_detail)
    except TooManyRedirects as e:
        error_detail = f"Too Many Redirects [aitools] - Error: {e}"
        logger.error(f"TooManyRedirects encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=310, detail=error_detail)
    except InvalidURL as e:
        error_detail = f"Invalid URL [aitools] - Error: {e}"
        logger.error(f"InvalidURL encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=error_detail)
    except RequestError as e:
        await client_manager.increment_error()
        error_detail = f"Request Error [aitools] - Error: {e}"
        logger.error(f"RequestError encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)
    except Exception as e:
        await client_manager.increment_error()
        error_detail = f"Unknown Error [aitools] - Error: {e}"
        logger.error(f"Unknown Error encountered: {e}. Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)

    if rp_resp.status_code > 399:
        logger.error(f"Non-2xx/3xx status code from openai: {rp_resp.status_code}")

    return StreamingResponse(
        rp_resp.aiter_raw(),
        status_code=rp_resp.status_code,
        headers=rp_resp.headers,
        background=BackgroundTask(rp_resp.aclose)
    )
