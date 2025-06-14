import os
import base64
import numpy as np
import uvicorn
import traceback
import dotenv

from PIL import Image
from io import BytesIO

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from models import LlavaRequest, LlavaResponse
from llava_server.llava import load_llava
from llava_server.bertscore import load_bertscore

from google import genai

os.environ["NUMEXPR_MAX_THREADS"] = "64"
dotenv.load_dotenv()

app = FastAPI()
# Create a dedicated directory for model offloading
offload_dir = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(offload_dir, exist_ok=True)

print("Loading LLaVA model...")
INFERENCE_FN = load_llava("liuhaotian/llava-v1.5-7b",
                          offload_folder=offload_dir,
                          device_map="auto")
print("LLaVA model loaded successfully!")

print("Loading BERTScore function...")
BERTSCORE_FN = load_bertscore()
print("BERTScore function loaded successfully!")

GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")


def extract_images_and_prompts(messages):
    """Extract images and prompts from messages in OpenAI format."""
    images = []
    queries = []

    for message in messages:
        if message.role == "user" and isinstance(message.content, list):
            # Extract images and text from this message
            prompt = ""
            for content_part in message.content:
                if content_part.get("type") == "text":
                    prompt += content_part.get("text", "")
                elif content_part.get("type") == "image_url":
                    image_url = content_part.get("image_url", {})
                    if "url" in image_url and image_url["url"].startswith("data:image"):
                        # Extract base64 data
                        base64_data = image_url["url"].split(",")[1]
                        image_data = base64.b64decode(base64_data)
                        images.append(image_data)
            queries.append(prompt)

    # Format as expected by INFERENCE_FN
    formatted_queries = [queries] * len(images)  # Each image gets all queries
    return images, formatted_queries


@app.post("/")
def inference(request: LlavaRequest):
    try:
        # Parse the OpenAI-like request format
        images_data, queries = extract_images_and_prompts(request.messages)

        # Convert image bytes to PIL Images
        images = [Image.open(BytesIO(img_data)) for img_data in images_data]

        print(
            f"Got {len(images)} images, {len(queries[0]) if queries else 0} queries per image")

        # Generate outputs from the model
        outputs = INFERENCE_FN(images, queries)

        # Prepare the OpenAI-like response
        response_data = {
            "model": request.model,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": output[0] if output else ""
                    },
                    "finish_reason": "stop"
                }
                for i, output in enumerate(outputs)
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

        # If answers are provided, calculate BERTScore
        if request.answers:
            print("Running bertscore...")
            output_shape = np.array(outputs).shape
            precision, recall, f1 = BERTSCORE_FN(
                np.array(outputs).reshape(-1).tolist(),
                np.array(request.answers).reshape(-1).tolist(),
            )

            response_data["bertscore"] = {
                "precision": precision.reshape(output_shape).tolist(),
                "recall": recall.reshape(output_shape).tolist(),
                "f1": f1.reshape(output_shape).tolist()
            }

        # Return the response as JSON
        return JSONResponse(response_data, status_code=200)

    except Exception:
        error = traceback.format_exc()
        print(error)
        return JSONResponse({"error": error}, status_code=500)


@app.post("/gemini")
def gemini_inference(request: LlavaRequest):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Parse the OpenAI-like request format
        images_data, queries = extract_images_and_prompts(request.messages)

        # Convert image bytes to PIL Images
        images = [Image.open(BytesIO(img_data)) for img_data in images_data]

        print(
            f"Got {len(images)} images, {len(queries[0]) if queries else 0} queries per image")

        # Upload images using google File API and send the request to Gemini
        outputs = []
        for image in images:
            image_bytes = BytesIO()
            image.save(image_bytes, format='PNG')
            image_bytes.seek(0)

            # Upload the image to Google Gemini
            image_file = client.files.upload(
                file=image_bytes,
                config={
                    "mime_type": "image/png",
                    "display_name": "image.png"
                }
            )

            # Generate Gemini response
            outputs.append(
                client.models.generate_content(
                    model=request.model,
                    contents=[image_file, queries[0]],
                ).text
            )

        # Prepare the OpenAI-like response
        response_data = {
            "model": request.model,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": output if output else ""
                    },
                    "finish_reason": "stop"
                }
                for i, output in enumerate(outputs)
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

        # If answers are provided, calculate BERTScore
        if request.answers:
            print("Running bertscore...")
            precision, recall, f1 = BERTSCORE_FN(
                outputs,
                np.array(request.answers).reshape(-1).tolist(),
            )

            response_data["bertscore"] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "f1": f1.tolist()
            }

        # Return the response as JSON
        return JSONResponse(response_data, status_code=200)

    except Exception:
        error = traceback.format_exc()
        print(error)
        return JSONResponse({"error": error}, status_code=500)


if __name__ == "__main__":
    print("Starting VLM server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
