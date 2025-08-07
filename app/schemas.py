from pydantic import BaseModel, ConfigDict, HttpUrl, validator
from datetime import datetime
from typing import Optional, List
import re

class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class ChatbotBase(BaseModel):
    organization_name: str
    chatbot_name: str

class ChatbotCreate(ChatbotBase):
    website_url: Optional[str] = None

    @validator('website_url')
    def validate_website_url(cls, v):
        if v is None or v.strip() == "":
            return None

        # Clean the URL
        v = v.strip()

        # Add protocol if missing
        if not v.startswith(('http://', 'https://')):
            v = 'https://' + v

        # Basic URL format validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        if not url_pattern.match(v):
            raise ValueError('Invalid URL format')

        return v

class Chatbot(ChatbotBase):
    id: int
    created_at: datetime
    owner_id: int
    website_url: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str 