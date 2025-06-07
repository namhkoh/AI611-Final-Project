from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import time


class ImageData(BaseModel):
    url: Optional[str] = None
    base64: Optional[str] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class LlavaRequest(BaseModel):
    model: str = "llava-v1.5-7b"
    messages: List[Message]
    max_tokens: int = 256
    temperature: float = 1.0
    answers: Optional[List[List[str]]] = None


class LlavaResponse(BaseModel):
    id: str = "chatcmpl-llava"
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    bertscore: Optional[Dict[str, List[List[float]]]] = None
