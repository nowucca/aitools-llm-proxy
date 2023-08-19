from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
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

# Configure logging based on VERBOSE_LOGGING
if VERBOSE_LOGGING:
    logging.basicConfig(level=logging.DEBUG)
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.DEBUG)
    httpcore_logger = logging.getLogger("httpcore")
    httpcore_logger.setLevel(logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)
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

    url = httpx.URL(path=api_path, query=request.url.query.encode("utf-8"))
    rp_req = client.build_request(request_method, url, timeout=20.0, headers=cleaned_headers, content=request_content)
    rp_resp = await client.send(rp_req, stream=True)

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
