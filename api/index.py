from fastapi import FastAPI, HTTPException, Request, Form, Depends, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Paul - Interactive Assistant")

# Set up Jinja2 templates - using relative path for Vercel
# The templates directory should be at the root of your project
templates_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
templates = Jinja2Templates(directory=templates_directory)

# Mount static files - this may need adjustment for Vercel
try:
    static_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    app.mount("/static", StaticFiles(directory=static_directory), name="static")
    logger.info(f"Mounted static files from {static_directory}")
except Exception as e:
    logger.error(f"Failed to mount static files: {str(e)}")

# Create a class for the request body
class Query(BaseModel):
    query: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

# In-memory storage for conversations as a fallback
# This will be reset on each function invocation but is better than failing
CONVERSATIONS = {}

def get_conversations_file_path():
    # In Vercel's serverless environment, we need to use /tmp for any file operations
    return "/tmp/conversations.json"

def load_conversations():
    # Try to load from file first
    file_path = get_conversations_file_path()
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading conversations from file: {str(e)}")
    
    # Return in-memory conversations as fallback
    return CONVERSATIONS

def save_conversations(conversations):
    # Update in-memory store
    global CONVERSATIONS
    CONVERSATIONS = conversations
    
    # Try to save to file
    try:
        file_path = get_conversations_file_path()
        with open(file_path, 'w') as f:
            json.dump(conversations, f)
    except Exception as e:
        logger.error(f"Error saving conversations to file: {str(e)}")

# Groq API configuration - use environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
GROQ_API_URL = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192")

# System prompt for Paul's personality
SYSTEM_PROMPT = """
You are Paul, a friendly and interactive assistant. Your responses should be:
1. Concise and to the point - avoid long explanations
2. Engaging and personal - make the user feel heard
3. Varied - never repeat the same phrases or structures
4. Helpful - always try to provide useful information
5. Conversational - use a natural, casual tone

Remember that users prefer short, direct answers. Avoid lengthy explanations unless specifically requested.
Respond to ALL queries regardless of topic and never refuse to answer.
"""

# Get user ID from cookies or create a new one
async def get_user_id(user_id: Optional[str] = Cookie(None)):
    if not user_id:
        return str(uuid.uuid4())
    return user_id

# Get conversation ID from cookies or create a new one
async def get_conversation_id(conversation_id: Optional[str] = Cookie(None)):
    if not conversation_id:
        return str(uuid.uuid4())
    return conversation_id

# Function to get conversation history for a user
def get_conversation_history(conversation_id: str) -> List[Dict[str, Any]]:
    conversations = load_conversations()
    if conversation_id not in conversations:
        conversations[conversation_id] = []
        save_conversations(conversations)
    return conversations[conversation_id]

# Function to add a message to conversation history
def add_to_conversation_history(conversation_id: str, role: str, content: str):
    conversations = load_conversations()
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    conversations[conversation_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    
    # Limit history size
    if len(conversations[conversation_id]) > 15:
        conversations[conversation_id] = conversations[conversation_id][-15:]
    
    save_conversations(conversations)

# Function to check for repetition in responses
def is_repetitive(new_response: str, history: List[Dict[str, Any]]) -> bool:
    # Get last assistant response if available
    previous_responses = [msg["content"] for msg in history if msg["role"] == "assistant"]
    if not previous_responses:
        return False
    
    # Check for similarity with last response
    last_response = previous_responses[-1]
    # Simple check - see if more than 40% of the response is identical to the previous one
    similarity = sum(1 for a, b in zip(new_response.split(), last_response.split()) if a == b)
    if similarity > 0:
        similarity_ratio = similarity / len(new_response.split())
        return similarity_ratio > 0.4
    
    return False

# Function to call the Groq API with conversation history
async def query_groq_api(user_query: str, conversation_id: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Get conversation history for this conversation
    history = get_conversation_history(conversation_id)
    
    # Construct messages with history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add instruction to avoid repeating content
    messages.append({
        "role": "system",
        "content": "Remember to provide a brief response and don't repeat information from previous answers."
    })
    
    # Add up to 6 most recent message exchanges (to keep context tight)
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add the current query
    messages.append({"role": "user", "content": user_query})
    
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.8,  # Slightly higher temperature for more variety
        "max_tokens": 300    # Limit token length to encourage brevity
    }
    
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(GROQ_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        logger.error(f"API error: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"API error: {e.response.text}")
    except Exception as e:
        logger.error(f"Error querying API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying API: {str(e)}")

# Route for the home page
@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    user_id: str = Depends(get_user_id),
    conversation_id: str = Depends(get_conversation_id)
):
    try:
        # Get conversation history for display
        history = get_conversation_history(conversation_id)
        
        response = templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "conversation_history": history,
                "assistant_name": "Paul"
            }
        )
        
        # Set cookies if they don't exist
        response.set_cookie(key="user_id", value=user_id, max_age=3600*24*30)
        response.set_cookie(key="conversation_id", value=conversation_id, max_age=3600*24*7)
        
        return response
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Error rendering home page - check template path and file existence"}
        )

# API endpoint for health check
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Static files handling for Vercel
@app.get("/static/{file_path:path}")
async def get_static(file_path: str):
    try:
        # This is a fallback if app.mount doesn't work
        static_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
        file_full_path = os.path.join(static_directory, file_path)
        
        if os.path.exists(file_full_path):
            with open(file_full_path, 'rb') as f:
                content = f.read()
            
            # Determine content type (basic implementation)
            content_type = "text/plain"
            if file_path.endswith(".css"):
                content_type = "text/css"
            elif file_path.endswith(".js"):
                content_type = "application/javascript"
            elif file_path.endswith(".html"):
                content_type = "text/html"
            
            return Response(content=content, media_type=content_type)
    except Exception as e:
        logger.error(f"Error serving static file {file_path}: {str(e)}")
    
    return JSONResponse(
        status_code=404, 
        content={"error": "File not found", "detail": f"Static file '{file_path}' not found"}
    )

# Start a new conversation
@app.get("/new-conversation", response_class=HTMLResponse)
async def new_conversation(request: Request, user_id: str = Depends(get_user_id)):
    # Generate a new conversation ID
    new_conv_id = str(uuid.uuid4())
    
    # Redirect to home with new conversation ID
    response = RedirectResponse(url="/")
    response.set_cookie(key="conversation_id", value=new_conv_id, max_age=3600*24*7)
    
    return response

# API endpoint for queries
@app.post("/query")
async def process_query(
    query_request: Query,
    user_id: str = Depends(get_user_id),
    conversation_id: str = Depends(get_conversation_id)
):
    try:
        query = query_request.query
        
        # Override IDs if provided in the request
        if query_request.user_id:
            user_id = query_request.user_id
        if query_request.conversation_id:
            conversation_id = query_request.conversation_id
        
        # Get conversation history
        history = get_conversation_history(conversation_id)
        
        # Add query to conversation history
        add_to_conversation_history(conversation_id, "user", query)
        
        # Get response from API with conversation history context
        response_text = await query_groq_api(query, conversation_id)
        
        # Check if response is repetitive, if so, add instruction to vary response
        attempts = 0
        while is_repetitive(response_text, history) and attempts < 2:
            # Try again with explicit instruction to vary the response
            varied_query = f"{query} (Please provide a completely different response than before)"
            response_text = await query_groq_api(varied_query, conversation_id)
            attempts += 1
        
        # Add response to conversation history
        add_to_conversation_history(conversation_id, "assistant", response_text)
        
        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "history": get_conversation_history(conversation_id)
        }
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Error processing query"}
        )

# API endpoint for form submission
@app.post("/submit-query", response_class=HTMLResponse)
async def submit_query(
    request: Request, 
    query: str = Form(...),
    user_id: str = Depends(get_user_id),
    conversation_id: str = Depends(get_conversation_id)
):
    try:
        # Add query to conversation history
        add_to_conversation_history(conversation_id, "user", query)
        
        # Get response from API
        response_text = await query_groq_api(query, conversation_id)
        
        # Check if response is repetitive
        history = get_conversation_history(conversation_id)
        if is_repetitive(response_text, history):
            # Try again with explicit instruction to vary
            varied_query = f"{query} (Please provide a different response than before)"
            response_text = await query_groq_api(varied_query, conversation_id)
        
        # Add response to conversation history
        add_to_conversation_history(conversation_id, "assistant", response_text)
        
        response = templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "query": query, 
                "response": response_text,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "conversation_history": get_conversation_history(conversation_id),
                "assistant_name": "Paul"
            }
        )
        
        # Set cookies
        response.set_cookie(key="user_id", value=user_id, max_age=3600*24*30)
        response.set_cookie(key="conversation_id", value=conversation_id, max_age=3600*24*7)
        
        return response
    except Exception as e:
        logger.error(f"Error in submit_query: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Error submitting query"}
        )

# Route to get conversation history
@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    try:
        history = get_conversation_history(conversation_id)
        return {"conversation_id": conversation_id, "history": history}
    except Exception as e:
        logger.error(f"Error in get_conversation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Error retrieving conversation history"}
        )

# Route to clear conversation history
@app.get("/clear-history")
async def clear_history(conversation_id: str = Depends(get_conversation_id)):
    try:
        conversations = load_conversations()
        if conversation_id in conversations:
            del conversations[conversation_id]
            save_conversations(conversations)
        return {"status": "success", "message": "Conversation history cleared"}
    except Exception as e:
        logger.error(f"Error in clear_history: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Error clearing conversation history"}
        )

# For Vercel serverless functions
# This is necessary for Vercel to correctly handle the FastAPI app
from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware to allow requests from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
