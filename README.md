# VoiceBot - Voice-Enabled AI Chatbot

A production-grade, voice-enabled chatbot using **Speech Recognition** and **Deep Learning** (LSTM-based intent classification). The system accepts voice input, converts it to text, processes it through a trained neural network, and generates intelligent responses.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (Browser)                │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ Web Speech   │  │  Chat UI     │  │ Text      │ │
│  │ API (Voice)  │──│  Display     │──│ Input     │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP POST /chat
                       ▼
┌─────────────────────────────────────────────────────┐
│              Backend (FastAPI + TensorFlow)          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │  Text        │  │  LSTM Model  │  │ Response  │ │
│  │  Preprocess  │──│  (Intent     │──│ Generator │ │
│  │              │  │  Classify)   │  │           │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────┘
```

## Tech Stack

| Component       | Technology                          |
|----------------|-------------------------------------|
| Speech Recognition | Web Speech API (Browser-native) |
| Deep Learning Model | TensorFlow/Keras LSTM (Bidirectional) |
| Backend Framework | FastAPI (Python)                  |
| Frontend        | HTML5 / CSS3 / Vanilla JavaScript  |
| Deployment      | Docker / Render.com / Railway      |

## Features

- **Voice Input**: Click the microphone button and speak naturally
- **Voice Output**: Bot responses are spoken aloud using Text-to-Speech (toggle on/off)
- **Speech-to-Text**: Real-time speech recognition using Web Speech API
- **Deep Learning**: Bidirectional LSTM model for intent classification
- **30 Intent Categories**: Greetings, jokes, time, weather, help, movies, books, travel, health, technology, and more
- **1500+ Training Patterns**: 50+ patterns per intent for robust classification
- **Confidence Scoring**: Shows model confidence for each response
- **Low Latency**: Optimized for fast speech recognition and response
- **Responsive UI**: Works on desktop and mobile
- **Production Ready**: Docker support, health checks, CORS enabled

## Project Structure

```
voice-chatbot/
├── backend/
│   ├── main.py              # FastAPI server
│   └── models/              # Trained model artifacts (generated during build)
│       ├── chatbot_model.keras  # (excluded from Git)
│       ├── tokenizer.pkl        # (excluded from Git)
│       ├── label_encoder.pkl    # (excluded from Git)
│       ├── responses.json
│       └── config.json
├── frontend/
│   └── index.html           # Web UI with voice input/output
├── training/
│   ├── train_model.py       # Model training script
│   └── data/
│       └── intents.json     # Training dataset (30 intents, 50+ patterns each)
├── .gitignore               # Excludes large model files from Git
├── Dockerfile               # Docker configuration (retrains model during build)
├── render.yaml              # Render.com deployment config
├── Procfile                 # Railway/Heroku deployment
├── requirements.txt         # Python dependencies
└── README.md
```

## Quick Start (Local Development)

### Prerequisites
- Python 3.9+
- pip

### Step 1: Install Dependencies

```bash
cd voice-chatbot
pip install -r requirements.txt
```

### Step 2: Train the Model (Local Only)

> **Note**: For deployment to Render, the model is automatically trained during Docker build. You only need to run this step for local development.

```bash
cd training
python train_model.py
```

This will:
- Load the intents dataset
- Tokenize and preprocess text
- Train a Bidirectional LSTM model
- Save model artifacts to `backend/models/`

Expected output:
```
Training Accuracy: ~0.95+
Test Accuracy: ~0.85+
Model saved to: backend/models/chatbot_model.keras
```

### Step 3: Run the Server

```bash
cd voice-chatbot
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 4: Open the App

Navigate to **http://localhost:8000** in your browser.

> **Note**: Speech recognition requires HTTPS or localhost. Use Chrome or Edge for best results.

## Deployment

### Important: Model Training

The trained model files (`.keras`, `.pkl`) are excluded from Git (via `.gitignore`) because they are too large for GitHub. Instead, the model is **automatically retrained during Docker build** on Render. This means:
- You only push source code and training data to GitHub
- Render retrains the model when building the Docker image (~2-3 minutes)
- No manual model training needed for deployment

### Option 1: Render.com (Recommended - Free)

1. Push this project to a GitHub repository
2. Go to [render.com](https://render.com) and sign up/login
3. Click **New > Web Service**
4. Connect your GitHub repository
5. Configure:
   - **Name**: `voice-chatbot`
   - **Runtime**: `Docker`
   - **Build Command**: (auto-detected from Dockerfile)
   - **Start Command**: (auto-detected from Dockerfile)
6. Click **Create Web Service**
7. Wait for deployment (5-10 minutes, includes model training)
8. Your live URL will be: `https://voice-chatbot.onrender.com`

### Option 2: Railway.app (Free Tier)

1. Push to GitHub
2. Go to [railway.app](https://railway.app)
3. Click **New Project > Deploy from GitHub repo**
4. Select your repository
5. Railway auto-detects the Dockerfile
6. Add a variable `PORT` = `8000` if needed
7. Deploy and get your live URL

### Option 3: Docker (Any Platform)

```bash
docker build -t voice-chatbot .
docker run -p 8000:8000 voice-chatbot
```

### Option 4: Manual Deployment (Any Python Host)

```bash
pip install -r requirements.txt
cd training && python train_model.py && cd ..
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint    | Description                    |
|--------|-------------|--------------------------------|
| GET    | `/`         | Serve the frontend UI          |
| POST   | `/chat`     | Send text, get bot response    |
| GET    | `/health`   | Health check                   |
| GET    | `/intents`  | List available intents         |

### Example Request

```bash
curl -X POST https://your-app-url.com/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}'
```

### Example Response

```json
{
  "intent": "mood",
  "response": "I'm doing great, thanks for asking! How can I help you today?",
  "confidence": 0.9523,
  "timestamp": "2026-09-28T12:00:00.000000"
}
```

## Supported Intents

| Intent | Example Phrases |
|--------|----------------|
| greeting | Hi, Hello, Good morning |
| goodbye | Bye, See you later |
| thanks | Thank you, I appreciate it |
| name | What is your name |
| help | Can you help me, What can you do |
| time | What time is it |
| date | What is today's date |
| weather | How's the weather |
| joke | Tell me a joke |
| age | How old are you |
| mood | How are you doing |
| creator | Who created you |
| compliment | You're awesome |
| insult | You're stupid |
| meaning_of_life | What is the meaning of life |
| math | Can you do math |
| music | What music do you like |
| food | What should I eat |
| programming | Best programming language |
| capabilities | What can you do |
| education | Help me study |
| sports | Do you like sports |
| movies | Recommend a movie |
| books | Suggest a book |
| travel | Where should I travel |
| health | How to stay healthy |
| technology | What is AI |
| hobby | Suggest a hobby |
| emergency | I need urgent help |
| fallback | (unrecognized input) |

## Model Details

- **Architecture**: Bidirectional LSTM with Embedding layer
- **Input**: Tokenized text sequences (max 20 tokens)
- **Output**: Softmax over 30 intent classes
- **Training**: Adam optimizer, early stopping, learning rate reduction
- **Vocabulary**: 1000 words
- **Embedding Dimension**: 64
- **Training Samples**: ~1500+
- **Training Accuracy**: ~89%
- **Test Accuracy**: ~56%

## Browser Compatibility

| Feature | Chrome | Edge | Firefox | Safari |
|---------|--------|------|---------|--------|
| Voice Input | Yes | Yes | No | No |
| Text Input | Yes | Yes | Yes | Yes |

> **Important**: Voice input uses the Web Speech API, which is supported in Chrome and Edge. For other browsers, users can type their messages.

## Troubleshooting

**Microphone not working?**
- Ensure you're on `localhost` or `HTTPS` (required for microphone access)
- Grant microphone permission when prompted
- Use Chrome or Edge browser

**Model not loading?**
- Run `python training/train_model.py` to generate model files
- Check that `backend/models/` contains all required files

**Deployment failing?**
- Check build logs for errors during model training
- Verify Python version compatibility (3.9+)
- Ensure `requirements.txt` includes all dependencies
- Render free tier has limited resources; model training may take 2-3 minutes

## License

This project is built for educational purposes as part of a Speech and Language Processing course.
