import os
import requests
import json
import base64
import argparse


def encode_image_to_base64(image_path):
    """Encode an image file as a base64 string"""
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_encoded = base64.b64encode(image_data).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_encoded}"


def test_vlm_server(image_path, prompt, server_url="http://localhost:8000"):
    """Test the Vision Language Model server with an image and prompt"""

    # Ensure image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Encode the image to base64
    base64_image = encode_image_to_base64(image_path)

    # Create the OpenAI-like payload
    payload = {
        "model": "gemini-2.0-flash",  # "llava-v1.5-7b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": base64_image},
                    },
                ],
            }
        ],
        "max_tokens": 256,
    }

    # Optional: Add answers for BERTScore calculation
    payload["answers"] = [["A monkey looking at the camera"]]

    try:
        # Send request to the server
        print(f"Sending request to {server_url}")
        print(f"Prompt: {prompt}")
        print(f"Image: {image_path}")
        print(f"Answers: {payload.get('answers', 'None')}")

        response = requests.post(server_url, json=payload)

        # Check response
        if response.status_code == 200:
            result = response.json()
            print("\nServer Response:")
            print(json.dumps(result, indent=2))

            # Print the model's answer
            if result.get("choices") and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"]
                print("\nModel's answer:", answer)

            # Print BERTScore if available
            if "bertscore" in result:
                print("\nBERTScore:")
                print(f"F1: {result['bertscore']['f1']}")
                print(f"Precision: {result['bertscore']['precision']}")
                print(f"Recall: {result['bertscore']['recall']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test VLM Server with an image and prompt"
    )
    parser.add_argument("--image", required=True,
                        help="Path to the image file")
    parser.add_argument(
        "--prompt",
        default="What is happening in this image?",
        help="Prompt for the VLM",
    )
    parser.add_argument(
        "--url", default="http://localhost:8000", help="URL of the VLM server"
    )

    args = parser.parse_args()

    test_vlm_server(args.image, args.prompt, args.url)
