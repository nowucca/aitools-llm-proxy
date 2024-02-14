import logging
from httpx import AsyncClient, Limits


class ClientManager:
    def __init__(self, base_url, timeout=60, error_threshold=10):
        self.base_url = base_url
        self.timeout = timeout
        self.error_threshold = error_threshold
        self.error_counter = 0
        self.limits = Limits(max_connections=200, max_keepalive_connections=20)
        self.client: AsyncClient = AsyncClient(base_url=self.base_url, timeout=self.timeout, limits=self.limits)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized HTTP client with base URL {self.base_url} and timeout {self.timeout} and limits {self.limits}")

    async def reset_client(self):
        try:
            if self.client:
                await self.client.aclose()
        except Exception as e:
            self.logger.error(f"Error closing HTTP client during reset: {e}")
        finally:
            self.logger.info(f"Resetting HTTP client after {self.error_counter}/{self.error_threshold} errors seen.")
            self.client = AsyncClient(base_url=self.base_url, timeout=self.timeout)
            self.error_counter = 0
            self.logger.info(f"Reset HTTP client with base URL {self.base_url} and timeout {self.timeout} and limits {self.limits}")

    async def increment_error(self):
        self.error_counter += 1
        self.logger.error(f"Incrementing error counter to {self.error_counter}/{self.error_threshold}")
        if self.error_counter >= self.error_threshold:
            await self.reset_client()

    async def get_client(self):
        return self.client

    async def close(self):
        try:
            if self.client:
                await self.client.aclose()
        except Exception as e:
            self.logger.error(f"Error closing HTTP client: {e}")