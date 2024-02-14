# A reverse proxy for OpenAI API

This is a reverse proxy for OpenAI, and supports chunked as well as non-chunked responses.
This lets us handle the chunked responses from OpenAI, and also lets us use
the asynchronous OpenAI API from the client application as well as inside this proxy.


## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the server

```bash
source venv/bin/activate
python3 simpler_proxy.py
```

A sample .env file:
```bash
OPENAI_API_KEY=(from openai)
OPENAI_API_BASE_URL=https://api.openai.com
OPENAI_ORG=(from openai)
VERBOSE_LOGGING=false
```
