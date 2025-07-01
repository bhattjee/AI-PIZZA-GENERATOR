from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pymongo import MongoClient

app = FastAPI()

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client.test  # database name: test
collection = db.pizzas  # collection name: pizzas

@app.get("/api/sample")
def get_sample_pizza():
    doc = collection.find_one()  # get first pizza doc
    if not doc:
        return JSONResponse(status_code=404, content={"message": "No documents found"})
    
    # Convert ObjectId to str safely
    return JSONResponse(content=jsonable_encoder(doc))
