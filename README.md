# AI611 Final Project

## LLAVA Vision Language Model Server

The project includes a FastAPI server for the LLAVA vision-language model that can process images and respond to text queries about them.

### Installation

```bash
# Clone the repository (if you haven't already)
git clone <repository-url>
cd AI611-Final-Project

# Install dependencies for the LLAVA server
cd LLAVA
poetry install
cd ..
```

### Running the LLAVA Server

```bash
# Run the server using Python directly
cd LLAVA
poetry run python app.py

# Or use uvicorn explicitly
poetry run uvicorn LLAVA.app:app --host 0.0.0.0 --port 8000
```

### Testing the LLAVA Server

The server exposes an API endpoint compatible with OpenAI's Vision API format:

```bash
cd LLAVA

# Test with a local image
poetry run python test.py --image ./monkey.png --prompt "What do you see in this image?"

# Test the Gemini inference
poetry run python test.py --image ./monkey.png --prompt "What do you see in this image?" --url http://localhost:8000/gemini
```

The LLAVA server requires about 35GB of GPU VRAM to run efficiently.

## Server API Documentation

The LLAVA server accepts POST requests to the root endpoint (`/`) with a JSON payload matching the OpenAI Vision API format:

- Images are sent as base64-encoded strings within the messages
- Multiple images can be processed in a batch
- Optional "answers" field can be provided to calculate BERTScore metrics

For detailed API documentation, see the [LLAVA README.md](LLAVA/README.md).

## Project Structure

- `LLAVA/`: Contains the LLAVA model server implementation
  - `app.py`: FastAPI server implementation
  - `models.py`: Pydantic models for request/response validation
  - `test.py`: Script for testing the server
  - `llava_server/`: Core LLAVA model functionality
