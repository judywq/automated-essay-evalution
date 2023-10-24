import aiohttp
import asyncio
from aiolimiter import AsyncLimiter
from tenacity import retry, wait_fixed, stop_after_attempt

MAX_CONCURRENT_REQUESTS = 10  # Set the maximum number of concurrent requests
MAX_TOKENS_PER_MINUTE = 10000  # Set the maximum number of tokens per minute (example value)
RATE_LIMIT_PERIOD = 60  # Period in seconds for the rate limit (1 minute in this case)

# Define a retry decorator
@retry(wait=wait_fixed(1000), stop=stop_after_attempt(3))  # Retry 3 times with a 1-second delay between retries
async def fetch_data(url, session, limiter, semaphore):
    async with semaphore, session.get(url) as response:
        # Estimate the number of tokens in the response (this is just an example, adjust based on actual response)
        response_text = await response.text()
        estimated_tokens = len(response_text.split())  # Assumes tokens are space-separated

        # Handle rate limiting based on estimated tokens
        async with limiter.acquire(estimated_tokens):
            return await response.json()

async def main():
    urls = ['https://api.example.com/data{}'.format(i) for i in range(1, 101)]  # Example URLs

    limiter = AsyncLimiter(rate_limit=MAX_TOKENS_PER_MINUTE, period=RATE_LIMIT_PERIOD)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data(url, session, limiter, semaphore) for url in urls]
        results = await asyncio.gather(*tasks)

        # Process the results here
        for result in results:
            print(result)

if __name__ == '__main__':
    asyncio.run(main())
