# Stale Fruit Freshness Prediction API

This is a FastAPI backend service for predicting fruit freshness using a Swin Transformer model hosted on Hugging Face. It includes user authentication, image upload, and prediction history stored in MongoDB Atlas.

## Features

- User signup and signin with JWT authentication
- Upload images for freshness prediction
- Calls external Hugging Face models for classification and freshness inference
- Saves prediction history per user in MongoDB
- CORS enabled to connect with frontend on different origins

## Setup

1. Clone the repository

    ```bash
    git clone https://github.com/yourusername/yourrepo.git
    cd yourrepo
2. Create and activate a Python virtual environment

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # or
    venv\Scripts\activate  # Windows
3. Install dependencies

    ```bash
    pip install -r requirements.txt
4. Create a .env file in the root directory and add your environment variables:

    ```bash
    MONGO_DETAILS=your_mongodb_connection_string
    SECRET_KEY=your_jwt_secret_key
    HF_API_TOKEN=your_huggingface_api_token
5. Run the FastAPI server

    ```bash
    uvicorn main:app --reload

## API Endpoints

`POST /signup` - Register new user

`POST /signin` - Login to get access token

`POST /predict` - Upload image and get freshness prediction (requires auth)

`POST /save_prediction` - Save prediction history (requires auth)

`GET /history` - Get prediction history for logged-in user