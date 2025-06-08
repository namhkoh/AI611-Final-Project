# LLaVA Server

Serves LLaVA inference using a FastAPI HTTP server. Supports batched inference and caches the embeddings for each image in order to produce multiple responses per image more efficiently.

## Installation
Requires Python 3.10 or newer.

```bash
cd LLAVA

# Install dependencies using Poetry
poetry install
```

## Open Port
Open server port for communication with client applications.
```bash
# Open port 8000
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload

# Close port 8000
sudo firewall-cmd --remove-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

## Run FastAPI Server
```bash
# Run the FastAPI server using uvicorn
cd LLAVA
poetry run python app.py

# Or run with explicit uvicorn command
poetry run uvicorn app:app --host 0.0.0.0 --port 8000
```
-> GPU VRAM required : ~ 35GB

## Testing the API
You can test the LLAVA server using the provided test script:

```bash
# Test with a local image
poetry run python test.py --image ./monkey.png --prompt "What do you see in this image?"

# Test the Gemini inference
poetry run python test.py --image ./monkey.png --prompt "What do you see in this image?" --url http://localhost:8000/gemini
```

## API Format

The API uses an OpenAI-compatible format:

### Request

```json
{
  "model": "llava-v1.5-7b", //gemini-2.0-flash
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What's in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."
          }
        }
      ]
    }
  ],
  "max_tokens": 256
}
```

### Response

```json
{
  "model": "llava-v1.5-7b", //gemini-2.0-flash
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The image shows a cat sitting on a windowsill."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

When a request arrives at the LLAVA server, you will see logs showing the image count and query information.
