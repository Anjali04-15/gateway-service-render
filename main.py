from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Body
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from typing import List
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from jose import JWTError, jwt
import httpx
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import traceback

app = FastAPI()

load_dotenv()

FRONTEND_URL = "https://stalefruitdetection.vercel.app"
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_DETAILS = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_DETAILS)
db = client["StaleFruit"]
users_collection = db["users"]
history_collection = db["history"]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

class UserIn(BaseModel):
    username: str
    email: EmailStr
    password: str

class SignInModel(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    username: str
    email: EmailStr

class Token(BaseModel):
    access_token: str
    token_type: str

class PredictionResult(BaseModel):
    label: str
    confidence: float

class PredictionHistory(BaseModel):
    user_email: str
    image_url: str
    prediction: str
    confidence: float
    timestamp: datetime

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="signin")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_user_by_email(email: str):
    return await users_collection.find_one({"email": email})

async def authenticate_user(email: str, password: str):
    user = await get_user_by_email(email)
    if not user or not verify_password(password, user['password']):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await get_user_by_email(email)
    if user is None:
        raise credentials_exception
    return user

@app.post("/signup", response_model=UserOut)
async def signup(user: UserIn):
    if await get_user_by_email(user.email):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    user_obj = {
        "username": user.username,
        "email": user.email,
        "password": hashed_password
    }
    await users_collection.insert_one(user_obj)
    return {"username": user.username, "email": user.email}

@app.post("/signin", response_model=Token)
async def signin(credentials: SignInModel):
    user = await authenticate_user(credentials.email, credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(
        data={"sub": user['email']},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

VALID_CLASSES = ['apple', 'banana', 'orange', 'tomato', 'bitter gourd', 'capsicum']

async def call_classification_model(image_bytes: bytes) -> bool:
    classification_api_url = "https://anjali04-15-fastapi-classification-model.hf.space/classify"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            classification_api_url,
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=60
        )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Classification model inference failed: {response.text}")
    
    data = response.json()
    
    if "prediction" not in data:
        raise HTTPException(status_code=500, detail="Invalid response from classification model")

    predicted_label = data["prediction"].lower()

    return predicted_label != "unknown"

async def call_swin_model(image_bytes: bytes):
    swin_api_url = "https://anjali04-15-swin-inference-api.hf.space/predictstatus"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            swin_api_url,
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=60
        )
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Swin model inference failed: {response.text}")
    
    data = response.json()
    
    if "swin_prediction" not in data:
        raise HTTPException(status_code=500, detail="Invalid response from Swin model")

    return data["swin_prediction"]

@app.post("/predict")
async def predict(file: UploadFile = File(...), current_user=Depends(get_current_user)):
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use JPEG or PNG.")
        
        try:
            image_bytes = await file.read()

            is_fruit = await call_classification_model(image_bytes)
            if not is_fruit:
                return {
                    "status": "invalid",
                    "message": "Uploaded image is not a fruit or vegetable"
                }

            swin_prediction = await call_swin_model(image_bytes)

            fruit_class = None
            if swin_prediction and "label" in swin_prediction:
                fruit_class = swin_prediction["label"].split("_")[-1]

            return {
                "status": "success",
                "fruit_class": fruit_class,
                "swin_prediction": swin_prediction
            }
        except Exception as e:
            print("Exception in /predict:")
            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e)
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }

@app.post("/save_prediction")
async def save_prediction(history: PredictionHistory, current_user=Depends(get_current_user)):
    try:
        record = history.dict()
        record['user_email'] = current_user['email']
        result = await history_collection.insert_one(record)
        return {"status": "success", "id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save prediction: {str(e)}")

@app.get("/get_history")
async def get_history(current_user=Depends(get_current_user)):
    try:
        cursor = history_collection.find({"user_email": current_user['email']}).sort("timestamp", -1)
        history = []
        async for doc in cursor:
            doc['id'] = str(doc['_id'])
            doc.pop('_id')
            history.append(doc)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "StaleFruit API is running"}

@app.get("/")
async def root():
    return {
        "message": "StaleFruit API",
        "documentation": f"{FRONTEND_URL}/docs",
        "status": "running"
    }
