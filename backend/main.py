import os
import json
import random
import pickle
import logging
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

model = None
tokenizer = None
label_encoder = None
responses_map = None
config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, label_encoder, responses_map, config
    logger.info("Loading model and artifacts...")
    try:
        model_path = os.path.join(MODEL_DIR, "chatbot_model.keras")
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")

        with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
            tokenizer = pickle.load(f)
        logger.info("Tokenizer loaded")

        with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        logger.info("Label encoder loaded")

        with open(os.path.join(MODEL_DIR, "responses.json"), "r") as f:
            responses_map = json.load(f)
        logger.info("Responses loaded")

        with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
            config = json.load(f)
        logger.info("Config loaded")

        logger.info("All artifacts loaded. Server ready!")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Voice-Enabled Chatbot API",
    description="A production-grade voice-enabled chatbot using deep learning and speech recognition",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    intent: str
    response: str
    confidence: float
    timestamp: str


def predict_intent(text: str) -> tuple:
    global model, tokenizer, label_encoder, responses_map, config

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=config["max_len"], padding="post")

    prediction = model.predict(padded, verbose=0)
    confidence = float(np.max(prediction))
    predicted_index = int(np.argmax(prediction))
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    if confidence < 0.5:
        predicted_label = "fallback"
        confidence = 1.0 - confidence

    if predicted_label == "time":
        response = datetime.now().strftime("The current time is %I:%M %p on %A, %B %d, %Y.")
    elif predicted_label == "date":
        response = f"Today's date is {datetime.now().strftime('%A, %B %d, %Y')}."
    else:
        possible_responses = responses_map.get(predicted_label, responses_map.get("fallback", ["I'm not sure how to respond to that."]))
        response = random.choice(possible_responses)

    return predicted_label, response, confidence


@app.get("/")
async def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(content={
        "status": "running",
        "message": "Voice Chatbot API is running. Access the frontend at /index.html",
        "endpoints": {
            "POST /chat": "Send text to get a response",
            "GET /health": "Health check",
            "GET /intents": "List available intents"
        }
    })


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        intent, response, confidence = predict_intent(request.text.strip())
        return ChatResponse(
            intent=intent,
            response=response,
            confidence=round(confidence, 4),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/intents")
async def list_intents():
    return {
        "intents": config["labels"],
        "count": len(config["labels"]),
        "training_accuracy": config.get("training_accuracy"),
        "test_accuracy": config.get("test_accuracy")
    }


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
