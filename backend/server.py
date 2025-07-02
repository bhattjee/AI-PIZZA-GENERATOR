from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
import uvicorn
from typing import List, Optional, Dict, Any, Union
from fastapi.encoders import jsonable_encoder
import os
import datetime
import requests
import json
import uuid
import re
from urllib.parse import quote_plus
from datetime import datetime
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
import logging
from fastapi.responses import JSONResponse
from datetime import timezone
import asyncio
import pymongo
import certifi
import ssl

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Pizza Recipe Generator API",
             description="API for generating custom pizza recipes using AI",
             version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://ai-pizza-generator.vercel.app",
        "https://pizzacrust.onrender.com"
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Validate environment variables
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY")
if not LLAMA_API_KEY:
    logger.error("LLAMA_API_KEY not found in environment variables")
    raise ValueError("LLAMA_API_KEY is required")

LLAMA_API_BASE_URL = os.environ.get(
    "LLAMA_API_BASE_URL",
    "https://openrouter.ai/api/v1"
)

LLAMA_TIMEOUT = int(os.environ.get("LLAMA_TIMEOUT", 30))  # seconds

MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    raise ValueError("MONGODB_URI environment variable not set")

if any(x in MONGO_URI.lower() for x in ["localhost", "127.0.0.1"]):
    raise ValueError(
        "MongoDB URI points to localhost - should be Atlas cluster or external MongoDB")

logger.info(
    f"MongoDB URI configured: {MONGO_URI.split('@')[-1].split('/')[0] if '@' in MONGO_URI else 'Invalid URI format'}")

# Initialize global variables for MongoDB
client = None
db = None
recipes_collection = None
sessions_collection = None

class MongoDBManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.recipes_collection = None
        self.sessions_collection = None
        self._connection_initialized = False

    async def initialize_connection(self):
        if self._connection_initialized and self.client:
            try:
                await asyncio.wait_for(
                    self.client.admin.command('ping'),
                    timeout=5.0
                )
                return self.client
            except Exception:
                logger.warning("Existing connection failed, reinitializing...")
                self._connection_initialized = False

        MONGO_URI = os.environ.get("MONGODB_URI")
        if not MONGO_URI:
            raise ValueError("MONGODB_URI environment variable not set")

        connection_params = {
            "connectTimeoutMS": 30000,
            "socketTimeoutMS": 30000,
            "serverSelectionTimeoutMS": 30000,
            "maxPoolSize": 10,
            "minPoolSize": 1,
            "maxIdleTimeMS": 60000,
            "waitQueueTimeoutMS": 15000,
            "tls": True,
            "tlsCAFile": certifi.where(),
            "tlsAllowInvalidCertificates": False,
            "tlsAllowInvalidHostnames": False,
            "retryWrites": True,
            "retryReads": True,
            "w": "majority",
            "readPreference": "primary",
            "heartbeatFrequencyMS": 10000,
            "maxConnecting": 2,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Attempting MongoDB connection (attempt {attempt + 1}/{max_retries})")

                self.client = AsyncIOMotorClient(
                    MONGO_URI, **connection_params)

                await asyncio.wait_for(
                    self.client.admin.command('ping'),
                    timeout=15.0
                )

                self.db = self.client.pizza_generator
                self.recipes_collection = self.db.recipes
                self.sessions_collection = self.db.user_sessions

                global client, db, recipes_collection, sessions_collection
                client = self.client
                db = self.db
                recipes_collection = self.recipes_collection
                sessions_collection = self.sessions_collection

                self._connection_initialized = True
                logger.info("MongoDB connection established successfully")
                return self.client

            except asyncio.TimeoutError:
                logger.warning(f"Connection attempt {attempt + 1} timed out")
            except pymongo.errors.ServerSelectionTimeoutError as e:
                logger.warning(
                    f"Server selection timeout on attempt {attempt + 1}: {str(e)[:200]}...")
            except Exception as e:
                logger.warning(
                    f"Connection attempt {attempt + 1} failed: {str(e)[:200]}...")

            if self.client:
                self.client.close()
                self.client = None

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)

        raise pymongo.errors.ServerSelectionTimeoutError(
            "Failed to connect to MongoDB after all retries")

    async def ensure_connection(self):
        if not self._connection_initialized or not self.client:
            await self.initialize_connection()
            return

        try:
            await asyncio.wait_for(
                self.client.admin.command('ping'),
                timeout=5.0
            )
        except Exception:
            logger.warning("Lost MongoDB connection, reconnecting...")
            self._connection_initialized = False
            await self.initialize_connection()

    async def safe_operation(self, operation, *args, **kwargs):
        max_retries = 2

        for attempt in range(max_retries):
            try:
                await self.ensure_connection()
                return await operation(*args, **kwargs)

            except (pymongo.errors.ServerSelectionTimeoutError,
                    pymongo.errors.NetworkTimeout,
                    pymongo.errors.AutoReconnect) as e:
                logger.warning(
                    f"Database operation failed (attempt {attempt + 1}): {e}")

                if attempt < max_retries - 1:
                    self._connection_initialized = False
                    await asyncio.sleep(1)
                else:
                    raise HTTPException(
                        status_code=503,
                        detail="Database temporarily unavailable"
                    )

            except Exception as e:
                logger.error(f"Unexpected database error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Database operation failed"
                )

mongo_manager = MongoDBManager()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"},
    )

@app.get("/health")
async def health_check():
    try:
        await mongo_manager.ensure_connection()
        await mongo_manager.client.admin.command('ping', socketTimeoutMS=5000)
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)},
        )

class IngredientSelection(BaseModel):
    category: str
    ingredients: List[str]

class DietaryPreferences(BaseModel):
    diet_types: List[str] = []
    additional_preferences: List[str] = []
    spice_level: int = 0
    allergen_avoidance: List[str] = []

class RecipeRequest(BaseModel):
    session_id: str
    ingredients: List[str]
    dietary_preferences: DietaryPreferences
    recipe_type: str

class CookingStep(BaseModel):
    step_number: int
    title: str
    description: str
    duration_minutes: Optional[int] = None
    temperature: Optional[str] = None
    ingredients_used: List[str] = []
    equipment: List[str] = []

class NutritionInfo(BaseModel):
    calories: Optional[int] = None
    protein: Optional[float] = None
    fat: Optional[float] = None
    carbohydrates: Optional[float] = None
    fiber: Optional[float] = None
    sugar: Optional[float] = None

class SauceStep(BaseModel):
    step_number: int
    title: str
    description: str
    duration_minutes: Optional[int] = None
    ingredients_used: List[str] = []
    equipment: List[str] = []

class Recipe(BaseModel):
    id: str
    name: str
    ingredients: List[str]
    detailed_ingredients: List[Dict[str, Any]] = []
    dietary_info: List[str]
    spice_level: int
    prep_time: int
    cook_time: int
    total_time: int
    servings: int
    difficulty: str
    steps: List[CookingStep]
    sauce_preparation: List[Union[SauceStep, str]]
    tips: List[str]
    source_url: Optional[str] = None
    nutrition: Optional[NutritionInfo] = None
    cost_per_serving: Optional[float] = None

INGREDIENT_CATEGORIES = {
    "flours": [
        "All-purpose", "Bread", "Whole wheat", "Semolina", "Type 00", "Rye", "Spelt",
        "Einkorn", "Gluten-free blend", "Almond (GF)", "Coconut (V, GF)", "Chickpea (V, GF)",
        "Buckwheat (V, GF)", "Rice (V, GF)", "Cornmeal (V, GF)", "Oat (V, GF)",
        "Quinoa (V, GF)", "Amaranth (V, GF)", "Teff (V, GF)", "Sorghum (V, GF)"
    ],
    "cheeses": [
        "Mozzarella", "Cheddar", "Provolone", "Parmesan", "Gorgonzola", "Ricotta",
        "Feta", "Goat cheese", "Pecorino romano", "Fontina", "Asiago", "Gouda",
        "Blue cheese", "Brie", "Camembert", "Edam", "Havarti", "Swiss",
        "Mascarpone", "Vegan cheese (V)"
    ],
    "meats": [
        "Pepperoni", "Italian sausage", "Spicy sausage", "Bacon", "Ham",
        "Grilled chicken", "Shredded chicken", "Ground beef", "Salami", "Prosciutto",
        "Anchovies", "Meatballs", "Canadian bacon", "Pancetta", "Soppressata",
        "Capicola", "Duck", "Turkey", "Chorizo", "Pastrami"
    ],
    "vegetables": [
        "Tomatoes", "Cherry tomatoes", "Red bell peppers", "Green bell peppers",
        "Yellow bell peppers", "Red onions", "White onions", "Yellow onions",
        "Button mushrooms", "Cremini mushrooms", "Shiitake mushrooms",
        "Black olives", "Green olives", "Spinach", "Artichoke hearts", "Arugula",
        "Zucchini", "Eggplant", "Corn", "Jalapeños"
    ],
    "sauces": [
        "Classic marinara (V)", "Tomato basil (V)", "Spicy arrabbiata (V)",
        "Tomato & roasted garlic (V)", "Alfredo", "Garlic cream", "Béchamel",
        "Basil pesto", "Sun-dried tomato pesto", "Spinach pesto", "Olive oil & garlic (V)",
        "Truffle oil (V)", "Chili oil (V)", "Barbecue (V)", "Buffalo", "Hummus (V)",
        "Tzatziki", "Peanut", "Mole", "Balsamic glaze"
    ],
    "spices_herbs": [
        "Oregano", "Dried basil", "Fresh basil", "Red pepper flakes", "Black pepper",
        "Garlic powder", "Onion powder", "Rosemary", "Thyme", "Parsley",
        "Italian seasoning", "Paprika", "Cumin", "Chili powder", "Cayenne",
        "Dill", "Sage", "Marjoram", "Bay leaves", "Cilantro"
    ],
    "other_toppings": [
        "Pine nuts", "Sun-dried tomatoes", "Capers", "Truffle slices",
        "Caramelized onions", "Roasted red peppers", "Avocado", "Pineapple",
        "Banana peppers", "Pickles", "Walnuts", "Almonds", "Raisins",
        "Cranberries", "Figs", "Pears", "Fresh buffalo mozzarella", "Honey", "Eggs"
    ]
}

def fix_json_response(response_text: str) -> Optional[Dict]:
    """Attempt to fix common JSON formatting issues in LLM responses"""
    try:
        # Remove JSON code block markers if present
        response_text = response_text.replace('```json', '').replace('```', '').strip()
        
        # If response is truncated, try to complete it
        if not response_text.endswith('}'):
            # Find the last complete object structure
            brace_count = 0
            last_valid_pos = 0
            
            for i, char in enumerate(response_text):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_valid_pos = i + 1
            
            if last_valid_pos > 0:
                response_text = response_text[:last_valid_pos]
            else:
                # Try to close the JSON properly
                open_braces = response_text.count('{') - response_text.count('}')
                open_brackets = response_text.count('[') - response_text.count(']')
                
                # Remove any trailing comma
                response_text = re.sub(r',\s*$', '', response_text.strip())
                
                # Close arrays first, then objects
                response_text += ']' * open_brackets
                response_text += '}' * open_braces
        
        # Fix common formatting issues
        # Remove trailing commas before closing brackets/braces
        response_text = re.sub(r',(\s*[}\]])', r'\1', response_text)
        
        # Fix unquoted keys
        response_text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', response_text)
        
        # Parse the cleaned JSON
        return json.loads(response_text)
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to fix JSON response: {e}")
        # If all else fails, try to extract a partial valid JSON
        try:
            # Look for the main recipe object
            start_idx = response_text.find('{')
            if start_idx != -1:
                # Try to find a reasonable end point
                for end_idx in range(len(response_text) - 1, start_idx, -1):
                    try:
                        partial_json = response_text[start_idx:end_idx + 1]
                        if partial_json.endswith('}'):
                            return json.loads(partial_json)
                    except:
                        continue
        except:
            pass
        return None
    except Exception as e:
        logger.error(f"Unexpected error in JSON fixing: {e}")
        return None

async def generate_recipe_with_llama(ingredients: List[str], dietary_preferences: DietaryPreferences) -> Optional[Dict]:
    """Generate recipe using OpenRouter API with improved JSON handling"""
    try:
        if not LLAMA_API_KEY:
            logger.error("LLAMA_API_KEY not configured")
            return None

        LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "deepseek/deepseek-r1-0528-qwen3-8b:free")

        # Create a more concise but complete prompt
        dietary_info = ", ".join(dietary_preferences.diet_types) if dietary_preferences.diet_types else "None"
        allergens = ", ".join(dietary_preferences.allergen_avoidance) if dietary_preferences.allergen_avoidance else "None"
        
        prompt = f"""Create a pizza recipe in JSON format:

Ingredients: {", ".join(ingredients[:8])}  
Diet: {dietary_info}
Avoid: {allergens}  
Spice: {dietary_preferences.spice_level}/4

Return valid JSON with this exact structure:
{{
  "name": "Creative pizza name",
  "ingredients": {ingredients[:8]},
  "dietary_info": {dietary_preferences.diet_types or ["Regular"]},
  "spice_level": {dietary_preferences.spice_level},
  "prep_time": 20,
  "cook_time": 15,
  "total_time": 35,
  "servings": 4,
  "difficulty": "Medium",
  "steps": [
    {{"step_number": 1, "title": "Make Dough", "description": "Mix flour, water, yeast, salt. Knead 8min. Rise 1hr.", "duration_minutes": 70, "ingredients_used": ["flour"], "equipment": ["bowl"]}},
    {{"step_number": 2, "title": "Prepare Sauce", "description": "Heat oil, add garlic, then sauce ingredients. Simmer 10min.", "duration_minutes": 10, "ingredients_used": ["sauce"], "equipment": ["pan"]}},
    {{"step_number": 3, "title": "Preheat Oven", "description": "Heat oven to 475°F. Place pizza stone if using.", "duration_minutes": 15, "temperature": "475°F", "equipment": ["oven"]}},
    {{"step_number": 4, "title": "Assemble Pizza", "description": "Roll dough, add sauce, cheese, then toppings.", "duration_minutes": 10, "ingredients_used": ["dough", "sauce", "toppings"], "equipment": ["rolling pin"]}},
    {{"step_number": 5, "title": "Bake", "description": "Bake 12-15min until golden and bubbly.", "duration_minutes": 15, "temperature": "475°F", "equipment": ["oven"]}}
  ],
  "sauce_preparation": [
    {{"step_number": 1, "title": "Heat Base", "description": "Heat 2 tbsp oil in pan over medium heat.", "duration_minutes": 2, "ingredients_used": ["oil"], "equipment": ["pan"]}},
    {{"step_number": 2, "title": "Add Aromatics", "description": "Add garlic, cook 1 minute until fragrant.", "duration_minutes": 1, "ingredients_used": ["garlic"], "equipment": []}},
    {{"step_number": 3, "title": "Build Sauce", "description": "Add selected sauce base and seasonings. Simmer 5-8 minutes.", "duration_minutes": 8, "ingredients_used": ["sauce base", "seasonings"], "equipment": []}}
  ],
  "tips": ["Use pizza stone for crispy crust", "Don't overload toppings", "Let dough rest at room temp"],
  "nutrition": {{"calories": 480, "protein": 18, "carbs": 52, "fat": 22}}
}}

Return ONLY the JSON, no other text."""

        payload = {
            "model": LLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "You are a pizza recipe expert. Return only properly formatted JSON responses with no additional text or markdown."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,  # Lower temperature for more consistent JSON
            "top_p": 0.8,
            "max_tokens": 2500,  # Increased token limit
            "response_format": {"type": "json_object"}
        }

        headers = {
            "Authorization": f"Bearer {os.environ.get('LLAMA_API_KEY')}",
            "HTTP-Referer": "https://ai-pizza-generator.vercel.app",
            "X-Title": "Pizzacraft-Key-M2",
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{LLAMA_API_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLAMA_TIMEOUT
        )

        if response.status_code == 401:
            logger.error("OpenRouter authentication failed. Check API key.")
            return None
        elif response.status_code == 429:
            logger.error("Rate limit exceeded. Please upgrade plan or wait.")
            return None
        elif response.status_code != 200:
            logger.error(f"API error {response.status_code}: {response.text}")
            return None

        try:
            response_data = response.json()
            if "choices" not in response_data or not response_data["choices"]:
                logger.error("Invalid API response structure")
                return None
                
            response_text = response_data["choices"][0]["message"]["content"].strip()
            
            # Log first 500 chars instead of 1000 to avoid log spam
            logger.info(f"LLM response preview: {response_text[:500]}{'...' if len(response_text) > 500 else ''}")

            # Try direct JSON parsing first
            try:
                recipe_data = json.loads(response_text)
                logger.info("Successfully parsed JSON response")
                return recipe_data
            except json.JSONDecodeError as e:
                logger.warning(f"Direct JSON parse failed: {e}")
                # Try to fix common JSON issues
                recipe_data = fix_json_response(response_text)
                if recipe_data:
                    logger.info("Successfully fixed and parsed JSON response")
                    return recipe_data
                else:
                    logger.error("Failed to fix JSON response")
                    return None

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse API response: {str(e)}")
            return None

    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in API call: {str(e)}")
        return None

def convert_llama_response_to_recipe(llama_response: Dict, ingredients: List[str], dietary_preferences: DietaryPreferences) -> Recipe:
    if not llama_response:
        return create_fallback_recipe(ingredients, dietary_preferences)

    steps = []
    for step_data in llama_response.get("steps", []):
        steps.append(CookingStep(
            step_number=step_data.get("step_number", 0),
            title=step_data.get("title", ""),
            description=step_data.get("description", ""),
            duration_minutes=step_data.get("duration_minutes"),
            temperature=step_data.get("temperature"),
            ingredients_used=step_data.get("ingredients_used", []),
            equipment=step_data.get("equipment", [])
        ))

    # FIX: Convert sauce preparation to plain dictionaries instead of Pydantic objects
    sauce_preparation = []
    raw_sauce = llama_response.get("sauce_preparation", [])
    
    if raw_sauce and isinstance(raw_sauce, list):
        for i, item in enumerate(raw_sauce):
            if isinstance(item, dict):
                # Create plain dictionary instead of SauceStep object
                sauce_preparation.append({
                    "step_number": item.get("step_number", i + 1),
                    "title": item.get("title", f"Sauce Step {i + 1}"),
                    "description": item.get("description", ""),
                    "duration_minutes": item.get("duration_minutes", 5),
                    "ingredients_used": item.get("ingredients_used", []),
                    "equipment": item.get("equipment", [])
                })
            elif isinstance(item, str):
                # Create plain dictionary for string items too
                sauce_preparation.append({
                    "step_number": i + 1,
                    "title": f"Sauce Step {i + 1}",
                    "description": item,
                    "duration_minutes": 5,
                    "ingredients_used": [],
                    "equipment": []
                })

    nutrition_data = llama_response.get("nutrition", {})
    nutrition = NutritionInfo(
        calories=nutrition_data.get("calories"),
        protein=nutrition_data.get("protein"),
        fat=nutrition_data.get("fat"),
        carbohydrates=nutrition_data.get("carbs"),
    ) if nutrition_data else None

    return Recipe(
        id=str(uuid.uuid4()),
        name=llama_response.get("name", "Custom Pizza"),
        ingredients=llama_response.get("ingredients", ingredients),
        dietary_info=llama_response.get("dietary_info", dietary_preferences.diet_types),
        spice_level=llama_response.get("spice_level", dietary_preferences.spice_level),
        prep_time=llama_response.get("prep_time", 15),
        cook_time=llama_response.get("cook_time", 20),
        total_time=llama_response.get("total_time", 35),
        servings=llama_response.get("servings", 4),
        difficulty=llama_response.get("difficulty", "Medium"),
        steps=steps,
        sauce_preparation=sauce_preparation,  # Now contains plain dicts
        tips=llama_response.get("tips", []),
        nutrition=nutrition
    )

def create_fallback_recipe(ingredients: List[str], dietary_preferences: DietaryPreferences) -> Recipe:
    has_meat = any(
        ingredient in INGREDIENT_CATEGORIES["meats"] for ingredient in ingredients)
    has_veggies = any(
        ingredient in INGREDIENT_CATEGORIES["vegetables"] for ingredient in ingredients)

    recipe_name = ("Hearty Meat & Veggie Pizza" if has_meat and has_veggies else
                  "Savory Meat Lovers Pizza" if has_meat else
                  "Garden Fresh Veggie Pizza" if has_veggies else
                  "Custom Artisan Pizza")

    steps = [
        CookingStep(
            step_number=1,
            title="Prepare the Dough",
            description="Mix flour with warm water, yeast, salt, and olive oil. Knead for 8-10 minutes until smooth. Let rise for 1 hour.",
            duration_minutes=70,
            equipment=["mixing bowl", "measuring cups"]
        ),
        CookingStep(
            step_number=2,
            title="Prepare Sauce",
            description="Combine your selected sauce ingredients. For marinara: blend tomatoes, garlic, herbs, and seasonings.",
            duration_minutes=10,
            equipment=["saucepan", "wooden spoon"]
        ),
        CookingStep(
            step_number=3,
            title="Preheat Oven",
            description="Preheat oven to 475°F (245°C). If using a pizza stone, place it in the oven while preheating.",
            duration_minutes=15,
            temperature="475°F (245°C)",
            equipment=["oven", "pizza stone (optional)"]
        ),
        CookingStep(
            step_number=4,
            title="Roll Out Dough",
            description="On a floured surface, roll out dough to desired thickness. Transfer to pizza pan or parchment paper.",
            duration_minutes=5,
            equipment=["rolling pin", "pizza pan"]
        ),
        CookingStep(
            step_number=5,
            title="Add Sauce",
            description="Spread sauce evenly over dough, leaving a 1-inch border for the crust.",
            duration_minutes=2,
            equipment=["spoon or ladle"]
        ),
        CookingStep(
            step_number=6,
            title="Add Toppings",
            description="Layer cheese first, then add your selected toppings. Don't overload to ensure even cooking.",
            duration_minutes=5
        ),
        CookingStep(
            step_number=7,
            title="Bake Pizza",
            description="Bake for 12-15 minutes until crust is golden and cheese is bubbly and slightly browned.",
            duration_minutes=15,
            temperature="475°F (245°C)"
        ),
        CookingStep(
            step_number=8,
            title="Cool and Serve",
            description="Let cool for 2-3 minutes, then slice and serve hot. Enjoy your custom pizza!",
            duration_minutes=3,
            equipment=["pizza cutter"]
        )
    ]

    # FIX: Return plain dictionaries instead of SauceStep objects
    sauce_steps = [
        {
            "step_number": 1,
            "title": "Heat oil",
            "description": "Heat 2 tbsp olive oil in a pan",
            "duration_minutes": 2,
            "ingredients_used": ["olive oil"],
            "equipment": ["pan"]
        },
        {
            "step_number": 2,
            "title": "Add garlic",
            "description": "Add minced garlic and cook for 1 minute",
            "duration_minutes": 1,
            "ingredients_used": ["garlic"],
            "equipment": []
        },
        {
            "step_number": 3,
            "title": "Add base",
            "description": "Add your selected sauce base and simmer for 5-10 minutes",
            "duration_minutes": 10,
            "ingredients_used": ["sauce base"],
            "equipment": []
        },
        {
            "step_number": 4,
            "title": "Season",
            "description": "Season with salt, pepper, and herbs to taste",
            "duration_minutes": 2,
            "ingredients_used": ["salt", "pepper", "herbs"],
            "equipment": []
        }
    ]

    return Recipe(
        id=str(uuid.uuid4()),
        name=recipe_name,
        ingredients=ingredients,
        dietary_info=dietary_preferences.diet_types,
        spice_level=dietary_preferences.spice_level,
        prep_time=90,
        cook_time=15,
        total_time=105,
        servings=4,
        difficulty="Medium",
        steps=steps,
        sauce_preparation=sauce_steps,  # Now contains plain dicts
        tips=[
            "Use a pizza stone for crispier crust",
            "Don't overload with toppings",
            "Let dough come to room temperature before rolling",
            "Brush crust with olive oil for golden color"
        ]
    )

async def generate_llama_recipe(ingredients: List[str], dietary_preferences: DietaryPreferences) -> Recipe:
    try:
        llama_response = await generate_recipe_with_llama(ingredients, dietary_preferences)
        if llama_response:
            return convert_llama_response_to_recipe(llama_response, ingredients, dietary_preferences)
        return create_fallback_recipe(ingredients, dietary_preferences)
    except Exception as e:
        logger.error(f"Recipe generation failed: {e}")
        return create_fallback_recipe(ingredients, dietary_preferences)

async def safe_db_operation(operation, *args, **kwargs):
    return await mongo_manager.safe_operation(operation, *args, **kwargs)

@app.get("/")
async def root():
    return {"message": "Pizza Generator API", "status": "running"}

@app.get("/api/ingredients")
async def get_ingredients():
    return {"categories": INGREDIENT_CATEGORIES}

@app.get("/api/ingredients/{category}")
async def get_ingredients_by_category(category: str):
    if category not in INGREDIENT_CATEGORIES:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"category": category, "ingredients": INGREDIENT_CATEGORIES[category]}

@app.post("/api/check-conflicts")
async def check_conflicts(ingredients: List[str], dietary_preferences: DietaryPreferences):
    conflicts = []
    for ingredient in ingredients:
        for allergen in dietary_preferences.allergen_avoidance:
            if allergen.lower() in ingredient.lower():
                conflicts.append({
                    "ingredient": ingredient,
                    "conflict_type": "allergen",
                    "conflict_detail": f"Contains {allergen}"
                })

    if "vegan" in [d.lower() for d in dietary_preferences.diet_types]:
        non_vegan_ingredients = [
            i for i in ingredients
            if any(meat in i.lower() for meat in ["cheese", "meat", "bacon", "ham", "pepperoni"])
        ]
        conflicts.extend({
            "ingredient": i,
            "conflict_type": "diet",
            "conflict_detail": "Not vegan"
        } for i in non_vegan_ingredients)

    return {"conflicts": conflicts}

@app.post("/api/find-related-recipes")
async def find_related_recipes(ingredients: List[str]):
    related = []
    RELATED_RECIPES = [
        {
            "name": "Classic Margherita",
            "ingredients": ["Mozzarella", "Tomato basil (V)", "Fresh basil", "Type 00"],
            "match_count": 0
        },
        {
            "name": "Pepperoni Classic",
            "ingredients": ["Pepperoni", "Mozzarella", "Classic marinara (V)", "All-purpose"],
            "match_count": 0
        },
        {
            "name": "Veggie Supreme",
            "ingredients": ["Mozzarella", "Mushrooms", "Bell peppers", "Red onions", "Black olives"],
            "match_count": 0
        }
    ]

    for recipe in RELATED_RECIPES:
        match_count = len(set(ingredients) & set(recipe["ingredients"]))
        if match_count >= 2:
            recipe_copy = recipe.copy()
            recipe_copy["match_count"] = match_count
            recipe_copy["match_percentage"] = round(
                (match_count / len(recipe["ingredients"])) * 100)
            related.append(recipe_copy)

    related.sort(key=lambda x: x["match_count"], reverse=True)
    return {"related_recipes": related[:5]}

@app.on_event("startup")
async def startup_db_client():
    try:
        await mongo_manager.initialize_connection()
        logger.info("Successfully initialized MongoDB connection")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB connection: {e}")
        pass

async def startup():
    logger.info("Starting Pizza Generator API")
    logger.info(f"Using model: {os.getenv('LLAMA_MODEL')}")
    logger.info(f"API base URL: {LLAMA_API_BASE_URL}")

@app.post("/api/generate-recipe", response_model_exclude={"_id"})
async def generate_recipe(request: RecipeRequest):
    try:
        logger.info(
            f"Received generate recipe request: session_id='{request.session_id}' ingredients={request.ingredients} dietary_preferences={request.dietary_preferences} recipe_type='{request.recipe_type}'"
        )

        await mongo_manager.safe_operation(
            mongo_manager.sessions_collection.update_one,
            {"session_id": request.session_id},
            {"$set": {
                "session_id": request.session_id,
                "ingredients": request.ingredients,
                "dietary_preferences": request.dietary_preferences.dict(),
                "timestamp": datetime.utcnow()
            }},
            upsert=True
        )

        recipe = await generate_llama_recipe(
            request.ingredients,
            request.dietary_preferences
        )

        recipe_data = jsonable_encoder(recipe)
        recipe_data.update({
            "session_id": request.session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "api_source": "llama"
        })

        try:
            await mongo_manager.safe_operation(
                mongo_manager.recipes_collection.insert_one,
                recipe_data.copy()
            )
            logger.info("Successfully saved recipe to database")
        except Exception as e:
            logger.error(f"Failed to insert recipe into DB: {e}")

        recipe_data.pop("_id", None)

        return JSONResponse(
            status_code=200,
            content={
                "recipe": recipe_data,
                "status": "success",
                "api_source": recipe_data["api_source"]
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate-recipe: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/recipe/{recipe_id}")
async def get_recipe(recipe_id: str):
    recipe = await mongo_manager.safe_operation(
        mongo_manager.recipes_collection.find_one,
        {"id": recipe_id}
    )

    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    recipe.pop("_id", None)
    return JSONResponse(content={"recipe": recipe})

@app.get("/api/session/{session_id}/recipes")
async def get_session_recipes(session_id: str):
    try:
        recipes_cursor = mongo_manager.recipes_collection.find(
            {"session_id": session_id}
        ).sort("created_at", -1)

        recipes = await mongo_manager.safe_operation(
            recipes_cursor.to_list,
            length=50
        )

        for recipe in recipes:
            recipe.pop("_id", None)

        return JSONResponse(content={
            "session_id": session_id,
            "recipes": recipes,
            "count": len(recipes)
        })

    except Exception as e:
        logger.error(f"Error fetching session recipes: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to fetch session recipes")

@app.get("/api/recipes/recent")
async def get_recent_recipes(limit: int = 10):
    try:
        if limit > 50:
            limit = 50

        recipes_cursor = mongo_manager.recipes_collection.find(
            {},
            {"session_id": 0}
        ).sort("created_at", -1).limit(limit)

        recipes = await mongo_manager.safe_operation(
            recipes_cursor.to_list,
            length=limit
        )

        for recipe in recipes:
            recipe.pop("_id", None)

        return JSONResponse(content={
            "recipes": recipes,
            "count": len(recipes)
        })

    except Exception as e:
        logger.error(f"Error fetching recent recipes: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to fetch recent recipes")

@app.delete("/api/recipe/{recipe_id}")
async def delete_recipe(recipe_id: str):
    try:
        result = await mongo_manager.safe_operation(
            mongo_manager.recipes_collection.delete_one,
            {"id": recipe_id}
        )

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Recipe not found")

        return JSONResponse(content={
            "message": "Recipe deleted successfully",
            "recipe_id": recipe_id
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting recipe: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete recipe")

@app.post("/api/recipe/{recipe_id}/save")
async def save_recipe_to_favorites(recipe_id: str, session_id: str):
    try:
        recipe = await mongo_manager.safe_operation(
            mongo_manager.recipes_collection.find_one,
            {"id": recipe_id}
        )

        if not recipe:
            raise HTTPException(status_code=404, detail="Recipe not found")

        await mongo_manager.safe_operation(
            mongo_manager.recipes_collection.update_one,
            {"id": recipe_id},
            {"$set": {"is_favorite": True, "favorited_at": datetime.now(
                timezone.utc).isoformat()}}
        )

        return JSONResponse(content={
            "message": "Recipe saved to favorites",
            "recipe_id": recipe_id
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving recipe to favorites: {e}")
        raise HTTPException(status_code=500, detail="Failed to save recipe")

@app.get("/api/session/{session_id}/favorites")
async def get_favorite_recipes(session_id: str):
    try:
        recipes_cursor = mongo_manager.recipes_collection.find(
            {"session_id": session_id, "is_favorite": True}
        ).sort("favorited_at", -1)

        recipes = await mongo_manager.safe_operation(
            recipes_cursor.to_list,
            length=50
        )

        for recipe in recipes:
            recipe.pop("_id", None)

        return JSONResponse(content={
            "session_id": session_id,
            "favorite_recipes": recipes,
            "count": len(recipes)
        })

    except Exception as e:
        logger.error(f"Error fetching favorite recipes: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to fetch favorite recipes")

@app.get("/api/stats")
async def get_api_stats():
    try:
        total_recipes = await mongo_manager.safe_operation(
            mongo_manager.recipes_collection.count_documents,
            {}
        )

        total_sessions = await mongo_manager.safe_operation(
            mongo_manager.sessions_collection.count_documents,
            {}
        )

        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0)
        recipes_today = await mongo_manager.safe_operation(
            mongo_manager.recipes_collection.count_documents,
            {"created_at": {"$gte": today_start.isoformat()}}
        )

        popular_ingredients = ["Mozzarella", "Pepperoni",
                              "Mushrooms", "Bell peppers", "Tomato basil"]

        return JSONResponse(content={
            "total_recipes": total_recipes,
            "total_sessions": total_sessions,
            "recipes_today": recipes_today,
            "popular_ingredients": popular_ingredients,
            "api_status": "healthy"
        })

    except Exception as e:
        logger.error(f"Error fetching API stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch statistics"}
        )

@app.post("/api/feedback")
async def submit_feedback(feedback_data: dict):
    try:
        feedback_entry = {
            "id": str(uuid.uuid4()),
            "rating": feedback_data.get("rating"),
            "comment": feedback_data.get("comment", ""),
            "recipe_id": feedback_data.get("recipe_id"),
            "session_id": feedback_data.get("session_id"),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "type": "recipe_feedback"
        }

        feedback_collection = mongo_manager.db.feedback

        await mongo_manager.safe_operation(
            feedback_collection.insert_one,
            feedback_entry
        )

        return JSONResponse(content={
            "message": "Feedback submitted successfully",
            "feedback_id": feedback_entry["id"]
        })

    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to submit feedback")

@app.delete("/api/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    try:
        recipes_result = await mongo_manager.safe_operation(
            mongo_manager.recipes_collection.delete_many,
            {"session_id": session_id}
        )

        session_result = await mongo_manager.safe_operation(
            mongo_manager.sessions_collection.delete_one,
            {"session_id": session_id}
        )

        return JSONResponse(content={
            "message": "Session cleaned up successfully",
            "deleted_recipes": recipes_result.deleted_count,
            "deleted_session": session_result.deleted_count > 0
        })

    except Exception as e:
        logger.error(f"Error cleaning up session: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to cleanup session")

@app.on_event("shutdown")
async def shutdown_db_client():
    try:
        if mongo_manager.client:
            mongo_manager.client.close()
            logger.info("MongoDB connection closed")
    except Exception as e:
        logger.error(f"Error closing MongoDB connection: {e}")

@app.middleware("http")
async def db_connection_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except pymongo.errors.ServerSelectionTimeoutError:
        logger.error("Database connection timeout")
        return JSONResponse(
            status_code=503,
            content={"detail": "Database temporarily unavailable"}
        )
    except Exception as e:
        logger.error(f"Unexpected error in middleware: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )