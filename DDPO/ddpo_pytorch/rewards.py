from PIL import Image
import io
import traceback
import requests
import os
import base64
import numpy as np
import torch
from typing import Iterable, List
from .bert_score import BERTScorer


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0,
                                                  255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from ddpo_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def llava_strict_satisfaction():
    """Submits images to LLaVA and computes a reward by matching the responses to ground truth answers directly without
    using BERTScore. Prompt metadata must have "questions" and "answers" keys. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 4
    url = "http://143.248.158.188:8000"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0,
                                                  255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(
            images, np.ceil(len(images) / batch_size))
        metadata_batched = np.array_split(
            metadata, np.ceil(len(metadata) / batch_size))

        all_scores = []
        all_info = {
            "answers": [],
        }

        for image_batch, metadata_batch in zip(images_batched, metadata_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [m["questions"] for m in metadata_batch],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            correct = np.array(
                [
                    [ans in resp for ans, resp in zip(m["answers"], responses)]
                    for m, responses in zip(metadata_batch, response_data["outputs"])
                ]
            )
            scores = correct.mean(axis=-1)

            all_scores += scores.tolist()
            all_info["answers"] += response_data["outputs"]

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn


def llava_bertscore():
    """Submits images to LLaVA and computes a reward by comparing the responses to the prompts using BERTScore. See
    https://github.com/kvablack/LLaVA-server for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://143.248.158.188:8000"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0,
                                                  255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(
            images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(
            prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            # scores = np.array(response_data["recall"]).squeeze()
            # 1) np.array(...) 자체가 1차원 배열이 되도록 하고, 절대 스칼라가 되지 않게 ravel() 사용
            scores = np.array(response_data["recall"], dtype=float).ravel()
            # 이제 scores는 항상 (batch,) 모양의 ndarray → .tolist()는 리스트를 반환
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            # all_info["precision"] += (np.array(response_data["precision"]).squeeze().tolist())
            # all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            # all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

            all_info["precision"] += np.array(
                response_data["precision"], dtype=float).ravel().tolist()
            all_info["f1"] += np.array(response_data["f1"],
                                       dtype=float).ravel().tolist()
            all_info["outputs"] += np.array(response_data["outputs"],
                                            dtype=object).ravel().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}
    return _fn


def llava_bertscore2():
    import os
    import pickle
    import dotenv

    from io import BytesIO
    from google import genai
    from google.genai import types

    dotenv.load_dotenv("../.env")

    batch_size = 16

    def load_bertscore():
        scorer = BERTScorer("microsoft/deberta-xlarge-mnli",
                            use_fast_tokenizer=True)
        print("BERT Loaded")

        def compute_bertscore(
            candidates: Iterable[str], references: Iterable[str]
        ) -> np.ndarray:
            precision, recall, f1 = scorer.score(candidates, references)
            return precision.numpy(), recall.numpy(), f1.numpy()

        return compute_bertscore

    def llava(image_path: str, prompt: str, endpoint: str = "https://silkworm-immortal-lively.ngrok-free.app/v1/chat/completions"):
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            b64 = base64.b64encode(image_bytes).decode("ascii")
            data_uri = f"data:image/jpeg;base64,{b64}"

            payload = {
                "model": "llava-hf/llava-1.5-7b-hf",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ]
                    }
                ]
            }

            headers = {"Content-Type": "application/json"}

            # Retry for a maximum of n times
            retry_limit = 3
            retry_count = 0
            while retry_count < retry_limit:
                response = requests.post(
                    endpoint, json=payload, headers=headers)

                if response.status_code != 200:
                    retry_count += 1
                    print(
                        f"Request failed with status {response.status_code}. Retrying {retry_count}/{retry_limit}...")
                    continue

                result = response.json()
                message = result['choices'][0]['message']['content']
                return message

            # Try gemini API after multiple failed attempts
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

            # Generate Gemini response
            return client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg",
                    ),
                    prompt,
                ],
            ).text
        except Exception as e:
            return f"Error: {e}\nRaw response: {response.text}"

    BERTSCORE_FN = load_bertscore()

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0,
                                                  255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(
            images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(
            prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]] * len(image_batch),
                "answers": [[f"The image contains {prompt}"] for prompt in prompt_batch],
            }
            data_bytes = pickle.dumps(data)

            print(f"received POST request")

            # send a request to the llava server
            retry = False
            while not retry:
                try:
                    # expects a dict with "images", "queries", and optionally "answers"
                    # images: (batch_size,) of JPEG bytes
                    # queries: (batch_size, num_queries_per_image) of strings
                    # answers: (batch_size, num_queries_per_image) of strings

                    data = pickle.loads(data_bytes)

                    images = [Image.open(BytesIO(d), formats=["jpeg"])
                              for d in data["images"]]
                    queries = data["queries"]

                    print(
                        f"Got {len(images)} images, {len(queries[0])} queries per image")
                    # queries = np.array(queries)  # (batch_size, num_queries_per_image)
                    # print("queries : ", queries)

                    save_dir = "./img_data"
                    os.makedirs(save_dir, exist_ok=True)
                    outputs = []
                    for i, img in enumerate(images):
                        path = os.path.join(save_dir, f"img{i}.jpg")
                        img.save(path)
                        # print(f"Saved image to {path}")

                        # print("query : ", queries[i][0])
                        output = llava(path, queries[i][0])
                        outputs.append(output)
                    # print("outputs : ", outputs)

                    #######################################################

                    response = {"outputs": outputs}

                    if "answers" in data:
                        print(f"Running bertscore...")
                        # print("inputs for bert", outputs)
                        output_shape = np.array(outputs).shape
                        (
                            response["precision"],
                            response["recall"],
                            response["f1"],
                        ) = BERTSCORE_FN(
                            np.array(outputs).reshape(-1).tolist(),
                            np.array(data["answers"]).reshape(-1).tolist(),
                        )

                        for key in ["precision", "recall", "f1"]:
                            response[key] = response[key].reshape(
                                output_shape).tolist()
                        # print("response1 : ", response)

                    # returns: a dict with "outputs" and optionally "scores"
                    # outputs: (batch_size, num_queries_per_image) of strings
                    # precision: (batch_size, num_queries_per_image) of floats
                    # recall: (batch_size, num_queries_per_image) of floats
                    # f1: (batch_size, num_queries_per_image) of floats
                    response = pickle.dumps(response)
                    # print("response2 : ", response)

                    returncode = 200
                except Exception as e:
                    response = traceback.format_exc()
                    print(response)
                    response = response.encode("utf-8")
                    returncode = 500

                if returncode == 200:
                    retry = True

            response_data = pickle.loads(response)
            # print("response data : ", response_data)
            # use the recall score as the reward
            # scores = np.array(response_data["recall"]).squeeze()
            # 1) np.array(...) 자체가 1차원 배열이 되도록 하고, 절대 스칼라가 되지 않게 ravel() 사용
            scores = np.array(response_data["recall"], dtype=float).ravel()
            # 이제 scores는 항상 (batch,) 모양의 ndarray → .tolist()는 리스트를 반환
            all_scores += scores.tolist()

            # save the precision and f1 scores for analysis
            # all_info["precision"] += (np.array(response_data["precision"]).squeeze().tolist())
            # all_info["f1"] += np.array(response_data["f1"]).squeeze().tolist()
            # all_info["outputs"] += np.array(response_data["outputs"]).squeeze().tolist()

            all_info["precision"] += np.array(
                response_data["precision"], dtype=float).ravel().tolist()
            all_info["f1"] += np.array(response_data["f1"],
                                       dtype=float).ravel().tolist()
            all_info["outputs"] += np.array(response_data["outputs"],
                                            dtype=object).ravel().tolist()

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn
