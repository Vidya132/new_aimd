from fastapi import FastAPI, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np

app = FastAPI()

# MongoDB connection URI (replace with your MongoDB URI)
MONGO_DB_URI = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_DB_URI)
db = client.my_database  # Use the database name you want

# Load the sentiment analysis model
model = tf.keras.models.load_model("model_tuned.h5")

# (Optional) Load any pre-processing steps from training history, if necessary
# with open("training_history.pkl", "rb") as file:
#     training_history = pickle.load(file)
# For example, if you had a tokenizer, you could load it as:
# tokenizer = training_history.get("tokenizer")  # Adjust as needed

# Define Pydantic models for input validation
class Item(BaseModel):
    name: str
    description: str = None

class TextInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI with MongoDB and Sentiment Analysis"}

@app.post("/analyze_sentiment/")
async def analyze_sentiment(input: TextInput):
    # Preprocess input text as needed (e.g., using tokenizer)
    text = input.text
    # Example: Tokenize and pad the text if your model requires it
    # sequences = tokenizer.texts_to_sequences([text])
    # padded_sequences = pad_sequences(sequences, maxlen=your_model_input_length)
    # prediction = model.predict(padded_sequences)
    
    # Here we're assuming model can take raw text as input; adjust as necessary
    try:
        prediction = model.predict([text])  # Replace with proper input processing if needed
        sentiment = "positive" if np.argmax(prediction) == 1 else "negative"
        return {"sentiment": sentiment, "score": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@app.post("/items/", response_model=dict)
async def create_item(item: Item):
    item_dict = item.dict()
    result = await db.my_collection.insert_one(item_dict)
    if result.inserted_id:
        return {"id": str(result.inserted_id), **item_dict}
    raise HTTPException(status_code=500, detail="Item could not be created")

@app.get("/items/")
async def get_items():
    items = await db.my_collection.find().to_list(100)  # Limits to 100 items
    return items


# from fastapi import FastAPI, HTTPException
# from motor.motor_asyncio import AsyncIOMotorClient
# from pydantic import BaseModel

# app = FastAPI()

# # MongoDB connection URI (replace with your MongoDB URI)
# MONGO_DB_URI = "mongodb://localhost:27017"
# client = AsyncIOMotorClient(MONGO_DB_URI)
# db = client.my_database  # Use the database name you want

# # Pydantic model to validate data
# class Item(BaseModel):
#     name: str
#     description: str = None

# @app.get("/")
# async def root():
#     return {"message": "Welcome to FastAPI with MongoDB"}

# @app.post("/items/", response_model=dict)
# async def create_item(item: Item):
#     item_dict = item.model_dump()
#     result = await db.my_collection.insert_one(item_dict)
#     if result.inserted_id:
#         return {"id": str(result.inserted_id), **item_dict}
#     raise HTTPException(status_code=500, detail="Item could not be created")

# @app.get("/items/")
# async def get_items():
#     items = await db.my_collection.find().to_list(100)  # Limits to 100 items
#     return items
