from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
import uvicorn
from typing import List, Optional, Dict, Any
from fastapi.encoders import jsonable_encoder
import os
import datetime
import requests
import json
import uuid
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

# Add this right after the imports
global client, db, recipes_collection, sessions_collection
client = None
db = None
recipes_collection = None
sessions_collection = None

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Correct Hugging Face inference API base URL for Mistral
LLAMA_API_BASE_URL = os.environ.get(
    "LLAMA_API_BASE_URL",
    "https://openrouter.ai/api/v1/chat/completions"
)

# Request timeout setting
LLAMA_TIMEOUT = 30  # seconds

# Validate environment variables
MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    raise ValueError("MONGODB_URI environment variable not set")

if any(x in MONGO_URI for x in ["localhost", "127.0.0.1"]):
    raise ValueError(
        "MongoDB URI points to localhost - should be Atlas cluster")
logger.info(
    f"Connecting to MongoDB at: {MONGO_URI.split('@')[-1].split('/')[0]}")

# Initialize MongoDB connection
def get_mongo_client():
    """Initialize MongoDB client with proper error handling"""
    try:
        # Initialize connection
        temp_client = AsyncIOMotorClient(
            MONGO_URI,
            tls=True,
            tlsCAFile=certifi.where(),
            connectTimeoutMS=10000,
            socketTimeoutMS=30000,
            serverSelectionTimeoutMS=10000,
            maxPoolSize=10,
            retryWrites=True,
            retryReads=True
        )

        # Assign to global variables
        global client, db, recipes_collection, sessions_collection
        client = temp_client
        db = client.pizza_generator
        recipes_collection = db.recipes
        sessions_collection = db.user_sessions
        logger.info("MongoDB client configured successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to configure MongoDB client: {e}")
        raise

# Global MongoDB client
try:
    client = get_mongo_client()
    db = client.get_database("pizza_generator")
    recipes_collection = db.recipes
    sessions_collection = db.user_sessions
    logger.info("MongoDB collections initialized")
except Exception as e:
    logger.error(f"Failed to initialize MongoDB: {e}")
    raise

# Create custom SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_REQUIRED

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"},
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        await client.admin.command('ping', socketTimeoutMS=5000)
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # Try to reconnect
        try:
            global client, db, recipes_collection, sessions_collection
            client = get_mongo_client()
            db = client.pizza_generator
            recipes_collection = db.recipes
            sessions_collection = db.user_sessions
            return {"status": "reconnected", "database": "connected"}
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"status": "unhealthy", "error": str(e)},
            )

# Pydantic models
class IngredientSelection(BaseModel):
    category: str
    ingredients: List[str]

class DietaryPreferences(BaseModel):
    diet_types: List[str] = []
    additional_preferences: List[str] = []
    spice_level: int = 0  # 0-4 scale
    allergen_avoidance: List[str] = []

class RecipeRequest(BaseModel):
    session_id: str
    ingredients: List[str]
    dietary_preferences: DietaryPreferences
    recipe_type: str  # "related" or "custom"

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
    sauce_preparation: List[str]
    tips: List[str]
    source_url: Optional[str] = None
    nutrition: Optional[NutritionInfo] = None
    cost_per_serving: Optional[float] = None

class MongoDBManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.recipes_collection = None
        self.sessions_collection = None

    async def connect(self):
        """Establish MongoDB connection with robust error handling"""
        MONGO_URI = os.environ.get(
            "MONGODB_URI") or os.environ.get("MONGO_URL")
        if not MONGO_URI:
            raise ValueError("MONGODB_URI environment variable not set")

        # Parse and reconstruct URI if needed
        if "mongodb+srv://" in MONGO_URI:
            # For MongoDB Atlas, ensure proper SSL parameters
            if "ssl=true" not in MONGO_URI and "tls=true" not in MONGO_URI:
                separator = "&" if "?" in MONGO_URI else "?"
                MONGO_URI += f"{separator}ssl=true&tlsAllowInvalidCertificates=false"

        connection_params = {
            # Core connection settings
            "connectTimeoutMS": 30000,  # Increased timeout
            "socketTimeoutMS": 30000,
            "serverSelectionTimeoutMS": 30000,  # Increased timeout
            "maxPoolSize": 10,
            "minPoolSize": 1,
            "maxIdleTimeMS": 45000,
            "waitQueueTimeoutMS": 10000,

            # SSL/TLS settings - FIXED
            "tls": True,
            "tlsCAFile": certifi.where(),
            "tlsAllowInvalidCertificates": False,
            "tlsAllowInvalidHostnames": False,

            # Retry settings
            "retryWrites": True,
            "retryReads": True,

            # Read/Write concerns
            "w": "majority",
            "readPreference": "primary",

            # Heartbeat settings
            "heartbeatFrequencyMS": 10000,

            # Connection management
            "maxConnecting": 2,
        }

        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Attempting MongoDB connection (attempt {attempt + 1}/{max_retries})")

                # Create client with connection parameters
                self.client = AsyncIOMotorClient(
                    MONGO_URI, **connection_params)

                # Test connection with admin command
                await asyncio.wait_for(
                    self.client.admin.command('ping'),
                    timeout=15.0
                )

                # Initialize database and collections
                self.db = self.client.pizza_generator
                self.recipes_collection = self.db.recipes
                self.sessions_collection = self.db.user_sessions

                logger.info(
                    f"MongoDB connection established successfully (attempt {attempt + 1})")
                return self.client

            except asyncio.TimeoutError:
                logger.warning(f"Connection attempt {attempt + 1} timed out")
                if self.client:
                    self.client.close()
                    self.client = None

            except pymongo.errors.ServerSelectionTimeoutError as e:
                logger.warning(
                    f"Server selection timeout on attempt {attempt + 1}: {str(e)[:200]}...")
                if self.client:
                    self.client.close()
                    self.client = None

            except pymongo.errors.ConfigurationError as e:
                logger.error(f"MongoDB configuration error: {e}")
                raise  # Don't retry configuration errors

            except ssl.SSLError as e:
                logger.warning(f"SSL error on attempt {attempt + 1}: {e}")
                if self.client:
                    self.client.close()
                    self.client = None

            except Exception as e:
                logger.warning(
                    f"Unexpected error on attempt {attempt + 1}: {str(e)[:200]}...")
                if self.client:
                    self.client.close()
                    self.client = None

            # Wait before retry (except on last attempt)
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)

        logger.error("All MongoDB connection attempts failed")
        raise pymongo.errors.ServerSelectionTimeoutError(
            "Failed to connect to MongoDB after all retries")

    async def ensure_connection(self):
        """Ensure we have a valid connection, reconnect if needed"""
        if not self.client:
            await self.connect()
            return

        try:
            # Quick ping to check connection
            await asyncio.wait_for(
                self.client.admin.command('ping'),
                timeout=5.0
            )
        except (Exception, asyncio.TimeoutError):
            logger.warning("Lost MongoDB connection, reconnecting...")
            if self.client:
                self.client.close()
            await self.connect()

    async def safe_operation(self, operation, *args, **kwargs):
        """Execute database operation with automatic reconnection"""
        max_retries = 3

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
                    # Force reconnection
                    if self.client:
                        self.client.close()
                        self.client = None
                    await asyncio.sleep(1)
                else:
                    raise

            except Exception as e:
                logger.error(f"Unexpected database error: {e}")
                raise

        raise pymongo.errors.ServerSelectionTimeoutError(
            "Database operation failed after all retries")

# Global MongoDB manager instance
mongo_manager = MongoDBManager()

# Ingredient categories data
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

# LLAMA API functions
async def generate_recipe_with_llama(ingredients: List[str], dietary_preferences: DietaryPreferences) -> Optional[Dict]:
    """Generate recipe using Hugging Face Inference API with Mistral-7B"""
    try:
        if not LLAMA_API_KEY:
            logger.error("LLAMA_API_KEY not configured")
            return None

        prompt = f"""Generate a detailed pizza recipe in strict JSON format with these requirements:

Input Parameters:
- Ingredients: {", ".join(ingredients)}
- Dietary Preferences: {dietary_preferences.diet_types or 'None'}
- Allergens to Avoid: {dietary_preferences.allergen_avoidance or 'None'}
- Spice Level: {dietary_preferences.spice_level}/4

Required JSON Structure:
{{
  "name": "Creative pizza name",
  "ingredients": ["list", "of", "ingredients"],
  "dietary_info": ["list", "of", "dietary", "tags"],
  "spice_level": 1,
  "prep_time": 15,
  "cook_time": 20,
  "total_time": 35,
  "servings": 4,
  "difficulty": "Easy|Medium|Hard",
  "steps": [
    {{
      "step_number": 1,
      "title": "Step title",
      "description": "Detailed instructions",
      "duration_minutes": 10,
      "temperature": "Optional",
      "ingredients_used": ["list"],
      "equipment": ["list"]
    }}
  ],
  "sauce_preparation": [
    {{
      "step_number": 1,
      "title": "Step title",
      "description": "Detailed sauce preparation instructions",
      "duration_minutes": 5,
      "ingredients_used": ["list of sauce ingredients"],
      "equipment": ["list of sauce equipment"]
    }}
  ],
  "tips": ["Helpful tip 1", "Helpful tip 2"],
  "nutrition": {{
    "calories": 500,
    "protein": 20,
    "carbs": 60,
    "fat": 15
  }}
}}"""

        payload = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful pizza recipe assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1500
        }

        headers = {
            "Authorization": f"Bearer {LLAMA_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-pizza-generator.vercel.app",
            "X-Title": "ai-pizza-generator"
        }

        response = requests.post(
            LLAMA_API_BASE_URL,
            headers=headers,
            json=payload,
            timeout=LLAMA_TIMEOUT
        )

        if response.status_code == 200:
            try:
                response_text = response.json(
                )["choices"][0]["message"]["content"]
                clean_json = response_text.replace(
                    '```json', '').replace('```', '').strip()
                recipe_data = json.loads(clean_json)
                return recipe_data
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse API response: {str(e)}")
                return None
        else:
            logger.error(f"API error {response.status_code}: {response.text}")
            return None

    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in API call: {str(e)}")
        return None

def convert_llama_response_to_recipe(llama_response: Dict, ingredients: List[str], dietary_preferences: DietaryPreferences) -> Recipe:
    """Convert LLAMA API response to our Recipe model"""
    if not llama_response:
        return create_fallback_recipe(ingredients, dietary_preferences)

    steps = []
    for step_data in llama_response.get("steps", []):
        step = CookingStep(
            step_number=step_data.get("step_number", 0),
            title=step_data.get("title", ""),
            description=step_data.get("description", ""),
            duration_minutes=step_data.get("duration_minutes"),
            temperature=step_data.get("temperature"),
            ingredients_used=step_data.get("ingredients_used", []),
            equipment=step_data.get("equipment", [])
        )
        steps.append(step)

    nutrition_data = llama_response.get("nutrition", {})
    nutrition = NutritionInfo(
        calories=nutrition_data.get("calories"),
        protein=nutrition_data.get("protein"),
        fat=nutrition_data.get("fat"),
        carbohydrates=nutrition_data.get("carbohydrates"),
        fiber=nutrition_data.get("fiber"),
        sugar=nutrition_data.get("sugar")
    ) if nutrition_data else None

    return Recipe(
        id=str(uuid.uuid4()),
        name=llama_response.get("name", "Custom Pizza"),
        ingredients=llama_response.get("ingredients", ingredients),
        dietary_info=llama_response.get(
            "dietary_info", dietary_preferences.diet_types),
        spice_level=llama_response.get(
            "spice_level", dietary_preferences.spice_level),
        prep_time=llama_response.get("prep_time", 20),
        cook_time=llama_response.get("cook_time", 15),
        total_time=llama_response.get("total_time", 35),
        servings=llama_response.get("servings", 4),
        difficulty=llama_response.get("difficulty", "Medium"),
        steps=steps,
        sauce_preparation=llama_response.get(
            "sauce_preparation", ["Mix sauce ingredients and simmer for 10 minutes"]),
        tips=llama_response.get("tips", [
            "Preheat your oven for best results",
            "Let the dough rest before stretching",
            "Don't overload with toppings"
        ]),
        nutrition=nutrition
    )

def create_fallback_recipe(ingredients: List[str], dietary_preferences: DietaryPreferences) -> Recipe:
    """Create a structured fallback recipe when API is unavailable"""
    has_meat = any(
        ingredient in INGREDIENT_CATEGORIES["meats"] for ingredient in ingredients)
    has_veggies = any(
        ingredient in INGREDIENT_CATEGORIES["vegetables"] for ingredient in ingredients)

    recipe_name = "Hearty Meat & Veggie Pizza" if has_meat and has_veggies else \
        "Savory Meat Lovers Pizza" if has_meat else \
        "Garden Fresh Veggie Pizza" if has_veggies else \
        "Custom Artisan Pizza"

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
        sauce_preparation=[
            "Heat 2 tbsp olive oil in a pan",
            "Add minced garlic and cook for 1 minute",
            "Add your selected sauce base and simmer for 5-10 minutes",
            "Season with salt, pepper, and herbs to taste"
        ],
        tips=[
            "Use a pizza stone for crispier crust",
            "Don't overload with toppings",
            "Let dough come to room temperature before rolling",
            "Brush crust with olive oil for golden color"
        ]
    )

async def generate_llama_recipe(ingredients: List[str], dietary_preferences: DietaryPreferences) -> Recipe:
    """Generate recipe using LLAMA API with fallback"""
    try:
        llama_response = await generate_recipe_with_llama(ingredients, dietary_preferences)
        return convert_llama_response_to_recipe(llama_response, ingredients, dietary_preferences) if llama_response \
            else create_fallback_recipe(ingredients, dietary_preferences)
    except Exception as e:
        logger.error(f"Recipe generation failed: {e}")
        return create_fallback_recipe(ingredients, dietary_preferences)

# Database safe operation handler
async def safe_db_operation(operation, *args, **kwargs):
    try:
        return await operation(*args, **kwargs)
    except pymongo.errors.ServerSelectionTimeoutError as e:
        logger.error(f"Database timeout: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        raise HTTPException(status_code=500, detail="Database error")

# API Routes
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

# Updated FastAPI startup event
@app.on_event("startup")
async def startup_db_client():
    """Initialize MongoDB connection on startup"""
    try:
        await client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB connection: {e}")
        raise

# Updated generate_recipe endpoint with fixed DB operations
@app.post("/api/generate-recipe")
async def generate_recipe(request: RecipeRequest):
    try:
        # Use the safe wrapper for all DB operations
        await safe_db_operation(
            sessions_collection.update_one,
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
                recipe_data
            )
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
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recipe/{recipe_id}")
async def get_recipe(recipe_id: str):
    recipe = await safe_db_operation(
        recipes_collection.find_one,
        {"id": recipe_id}
    )
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    recipe.pop("_id", None)
    return {"recipe": recipe}

@app.get("/api/recipes/session/{session_id}")
async def get_session_recipes(session_id: str):
    recipes = []
    cursor = recipes_collection.find({"session_id": session_id})
    async for recipe in cursor:
        recipe.pop("_id", None)
        recipes.append(recipe)
    return {"recipes": recipes}

@app.post("/api/save-cooking-progress")
async def save_cooking_progress(session_id: str, step_number: int, completed: bool):
    progress_data = {
        "session_id": session_id,
        "step_number": step_number,
        "completed": completed,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    await safe_db_operation(
        sessions_collection.update_one,
        {"session_id": session_id},
        {"$set": {f"cooking_progress.step_{step_number}": progress_data}},
        upsert=True
    )

    return {"status": "progress_saved"}

@app.get("/api/cooking-progress/{session_id}")
async def get_cooking_progress(session_id: str):
    session = await safe_db_operation(
        sessions_collection.find_one,
        {"session_id": session_id}
    )
    return {"progress": session.get("cooking_progress", {})} if session else {"progress": {}}

@app.post("/api/remove-ingredient")
async def remove_ingredient(session_id: str, ingredient: str):
    try:
        session = await safe_db_operation(
            sessions_collection.find_one,
            {"session_id": session_id}
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        current_ingredients = session.get("ingredients", [])
        if ingredient in current_ingredients:
            updated_ingredients = [
                i for i in current_ingredients if i != ingredient]

            await safe_db_operation(
                sessions_collection.update_one,
                {"session_id": session_id},
                {"$set": {"ingredients": updated_ingredients}}
            )

            return {"status": "removed", "remaining_ingredients": updated_ingredients}
        return {"status": "not_found", "message": "Ingredient not in current selection"}

    except Exception as e:
        logger.error(f"Failed to remove ingredient: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to remove ingredient: {str(e)}"
        )

@app.post("/api/validate-recipe")
async def validate_recipe(recipe: Recipe):
    errors = []
    if not recipe.name:
        errors.append("Recipe name is required")
    if not recipe.ingredients:
        errors.append("At least one ingredient is required")
    if not recipe.steps:
        errors.append("At least one cooking step is required")

    for i, step in enumerate(recipe.steps):
        if not step.description:
            errors.append(f"Step {i+1} is missing a description")

    if recipe.nutrition:
        if recipe.nutrition.calories is not None and recipe.nutrition.calories < 0:
            errors.append("Calories cannot be negative")
        if recipe.nutrition.protein is not None and recipe.nutrition.protein < 0:
            errors.append("Protein cannot be negative")

    return {"valid": not bool(errors), "errors": errors if errors else None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)