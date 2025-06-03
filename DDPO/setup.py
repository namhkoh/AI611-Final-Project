from setuptools import setup, find_packages

setup(
    name="ddpo-pytorch",
    version="0.0.1",
    packages=["ddpo_pytorch"],
    python_requires=">=3.10",           # I used python==3.10.16
    install_requires=[
        "ml-collections",
        "absl-py",
        "diffusers[torch]==0.18.0",
        "accelerate==0.17",
        "wandb",
        "pytorch==2.5.1"
        "torchvision==0.20.1",
        "pytorch-cuda==11.8"
        "inflect==6.0.4",
        "pydantic==1.10.9",
        "transformers==4.29.2",
        "huggingface-hub==0.15.1",
    ],
)
