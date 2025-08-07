from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, LargeBinary
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    chatbots = relationship("Chatbot", back_populates="owner")

class Chatbot(Base):
    __tablename__ = "chatbots"

    id = Column(Integer, primary_key=True, index=True)
    organization_name = Column(String)
    chatbot_name = Column(String)
    pdf_data = Column(LargeBinary)  # Store PDF file as binary
    vector_store_data = Column(LargeBinary)  # Store FAISS index as binary
    website_url = Column(String, nullable=True)  # Store website URL
    created_at = Column(DateTime, default=datetime.utcnow)
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="chatbots")