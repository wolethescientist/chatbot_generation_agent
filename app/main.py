from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import os
from datetime import timedelta
from typing import List, Optional
import asyncio
import json

from . import models, schemas, security
from .database import engine, get_db
from .chatbot import ChatbotProcessor

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Connexxion Agent API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
chatbot_processor = ChatbotProcessor()

# Dependency to get current user
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> models.User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = security.verify_token(token, credentials_exception)
    user = db.query(models.User).filter(models.User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

@app.post("/register", response_model=schemas.User)
async def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = security.get_password_hash(user.password)
    db_user = models.User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/token", response_model=schemas.Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/create_chatbot", response_model=schemas.Chatbot)
async def create_chatbot(
    organization_name: str = Form(...),
    chatbot_name: str = Form(...),
    pdf_file: UploadFile = File(None),
    website_url: str = Form(None),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    import time
    start_time = time.time()

    print(f"Starting chatbot creation for user {current_user.username}: {organization_name}/{chatbot_name}")
    print(f"PDF file provided: {pdf_file is not None}")
    print(f"Website URL provided: {website_url is not None and website_url.strip()}")

    # Validate that at least one content source is provided
    if not pdf_file and not (website_url and website_url.strip()):
        raise HTTPException(status_code=400, detail="Either a PDF file or website URL must be provided")

    # Validate PDF file if provided
    pdf_content = None
    if pdf_file:
        if not pdf_file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        pdf_content = await pdf_file.read()
        if not pdf_content:
            raise HTTPException(status_code=400, detail="Empty PDF file")
        print(f"PDF file read: {len(pdf_content)} bytes")

    # Create chatbot object from form data
    chatbot = schemas.ChatbotCreate(
        organization_name=organization_name,
        chatbot_name=chatbot_name,
        website_url=website_url
    )

    # Process content (PDF and/or website) and create vector store
    try:
        processing_start = time.time()
        pdf_binary, vector_store_binary, website_error = await chatbot_processor.process_combined_content_and_create_vector_store(
            pdf_data=pdf_content,
            website_url=website_url
        )
        processing_time = time.time() - processing_start
        print(f"Content processing completed in {processing_time:.2f} seconds")

        # Log website processing result
        if website_url and website_url.strip():
            if website_error:
                print(f"Website processing failed: {website_error}")
            else:
                print("Website content successfully processed and integrated")

    except Exception as e:
        print(f"Error processing content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing content: {str(e)}")

    # Save chatbot to database
    try:
        db_start = time.time()
        db_chatbot = models.Chatbot(
            organization_name=chatbot.organization_name,
            chatbot_name=chatbot.chatbot_name,
            pdf_data=pdf_binary,
            vector_store_data=vector_store_binary,
            website_url=chatbot.website_url,
            owner_id=current_user.id
        )
        db.add(db_chatbot)
        db.commit()
        db.refresh(db_chatbot)

        db_time = time.time() - db_start
        total_time = time.time() - start_time
        print(f"Database save completed in {db_time:.2f} seconds")
        print(f"Total chatbot creation time: {total_time:.2f} seconds")

        # Include website processing status in response if applicable
        if website_url and website_url.strip() and website_error:
            print(f"Note: Chatbot created successfully, but website processing had issues: {website_error}")

        return db_chatbot
    except Exception as e:
        db.rollback()
        print(f"Error saving chatbot to database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving chatbot to database: {str(e)}")

@app.post("/chat/{chatbot_id}")  # Temporarily removed response_model for debugging
async def chat_with_bot(
    chatbot_id: int,
    message: schemas.ChatMessage,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chatbot = db.query(models.Chatbot).filter(
        models.Chatbot.id == chatbot_id,
        models.Chatbot.owner_id == current_user.id
    ).first()
    
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    try:
        print(f"=== CHAT ENDPOINT START ===")
        print(f"Processing chat message: '{message.message}' for chatbot {chatbot_id}")
        print(f"User ID: {current_user.id}")
        print(f"Chatbot owner ID: {chatbot.owner_id}")

        response = await chatbot_processor.get_chatbot_response(
            chatbot.vector_store_data,
            message.message
        )
        print(f"Generated response type: {type(response)}")
        print(f"Generated response length: {len(str(response)) if response else 0}")
        print(f"Generated response: {response}")

        # Ensure response is a string
        if response is None:
            print("⚠️  Response is None, using fallback")
            response = "I'm sorry, I couldn't generate a response. Please try again."
        elif not isinstance(response, str):
            print(f"⚠️  Response is not string ({type(response)}), converting")
            response = str(response)

        # Ensure response is not empty
        if not response.strip():
            print("⚠️  Response is empty, using fallback")
            response = "I'm sorry, I couldn't generate a response. Please try again."

        result = {"response": response}
        print(f"Final result: {result}")
        print(f"Result type: {type(result)}")
        print(f"Result response type: {type(result['response'])}")
        print(f"Result JSON serializable test: {json.dumps(result)}")
        print(f"=== CHAT ENDPOINT END ===")
        return result
    except Exception as e:
        print(f"Error getting chatbot response: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting chatbot response: {str(e)}")

@app.get("/chatbots")
async def get_user_chatbots(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    chatbots = db.query(models.Chatbot).filter(models.Chatbot.owner_id == current_user.id).all()

    # Convert to dict format for JSON response
    chatbots_data = []
    for chatbot in chatbots:
        chatbots_data.append({
            "id": chatbot.id,
            "organization_name": chatbot.organization_name,
            "chatbot_name": chatbot.chatbot_name,
            "website_url": chatbot.website_url,
            "created_at": chatbot.created_at.isoformat() if chatbot.created_at else None,
            "owner_id": chatbot.owner_id
        })

    # Create response with cache-busting headers
    response = JSONResponse(content=chatbots_data)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response

@app.delete("/chatbots/{chatbot_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chatbot(
    chatbot_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chatbot = db.query(models.Chatbot).filter(
        models.Chatbot.id == chatbot_id,
        models.Chatbot.owner_id == current_user.id
    ).first()

    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")

    try:
        db.delete(chatbot)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting chatbot: {str(e)}")



@app.post("/public/chat/{chatbot_id}")
async def public_chat_with_bot(
    chatbot_id: int,
    message: schemas.ChatMessage,
    db: Session = Depends(get_db)
):
    """Public chat endpoint for embedded chatbots (no authentication required)"""
    chatbot = db.query(models.Chatbot).filter(models.Chatbot.id == chatbot_id).first()

    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")

    try:
        print(f"=== PUBLIC CHAT ENDPOINT START ===")
        print(f"Processing public chat message: '{message.message}' for chatbot {chatbot_id}")

        response = await chatbot_processor.get_chatbot_response(
            chatbot.vector_store_data,
            message.message
        )
        print(f"Generated response: {response}")

        # Ensure response is a string
        if response is None:
            response = "I'm sorry, I couldn't generate a response. Please try again."
        elif not isinstance(response, str):
            response = str(response)

        # Ensure response is not empty
        if not response.strip():
            response = "I'm sorry, I couldn't generate a response. Please try again."

        result = {"response": response}
        print(f"=== PUBLIC CHAT ENDPOINT END ===")
        return result
    except Exception as e:
        print(f"Error in public chat endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting chatbot response: {str(e)}")