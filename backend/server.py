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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

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
    "https://openrouter.ai/api/v1"  # ✅ Base URL only
)

# Request timeout setting
LLAMA_TIMEOUT = 30  # seconds

# Validate MongoDB URI
MONGO_URI = os.getenv("MONGODB_URI")
if not MONGO_URI:
    raise ValueError("MONGODB_URI environment variable not set")

# Additional validation to ensure we're not using localhost
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
        """Initialize MongoDB connection with robust error handling"""
        if self._connection_initialized and self.client:
            try:
                # Test existing connection
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

        # Ensure proper SSL/TLS configuration for Atlas
        connection_params = {
            # Core connection settings
            "connectTimeoutMS": 30000,
            "socketTimeoutMS": 30000,
            "serverSelectionTimeoutMS": 30000,
            "maxPoolSize": 10,
            "minPoolSize": 1,
            "maxIdleTimeMS": 60000,
            "waitQueueTimeoutMS": 15000,

            # SSL/TLS settings for Atlas
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
            "maxConnecting": 2,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Attempting MongoDB connection (attempt {attempt + 1}/{max_retries})")

                # Create client
                self.client = AsyncIOMotorClient(
                    MONGO_URI, **connection_params)

                # Test connection
                await asyncio.wait_for(
                    self.client.admin.command('ping'),
                    timeout=15.0
                )

                # Initialize database and collections
                self.db = self.client.pizza_generator
                self.recipes_collection = self.db.recipes
                self.sessions_collection = self.db.user_sessions

                # Set global variables for backward compatibility
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
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)

        raise pymongo.errors.ServerSelectionTimeoutError(
            "Failed to connect to MongoDB after all retries")

    async def ensure_connection(self):
        """Ensure we have a valid connection, reconnect if needed"""
        if not self._connection_initialized or not self.client:
            await self.initialize_connection()
            return

        try:
            # Quick ping to check connection
            await asyncio.wait_for(
                self.client.admin.command('ping'),
                timeout=5.0
            )
        except Exception:
            logger.warning("Lost MongoDB connection, reconnecting...")
            self._connection_initialized = False
            await self.initialize_connection()

    async def safe_operation(self, operation, *args, **kwargs):
        """Execute database operation with automatic reconnection"""
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
                    # Force reconnection
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


# Global MongoDB manager instance
mongo_manager = MongoDBManager()


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
        await mongo_manager.ensure_connection()
        await mongo_manager.client.admin.command('ping', socketTimeoutMS=5000)
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
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
    """Generate recipe using OpenRouter API"""
    try:
        if not LLAMA_API_KEY:
            logger.error("LLAMA_API_KEY not configured")
            return None

        # Define model at the start
        LLAMA_MODEL = os.environ.get(
            "LLAMA_MODEL", 
            "deepseek/deepseek-r1-0528-qwen3-8b:free"
        )

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
            "model": LLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful pizza recipe assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1500
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
            response_text = response.json()["choices"][0]["message"]["content"]
            clean_json = response_text.replace('```json', '').replace('```', '').strip()
            recipe_data = json.loads(clean_json)
            return recipe_data
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse API response: {str(e)}")
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
        if llama_response:
            return convert_llama_response_to_recipe(llama_response, ingredients, dietary_preferences)
        return create_fallback_recipe(ingredients, dietary_preferences)
    except Exception as e:
        logger.error(f"Recipe generation failed: {e}")
        return create_fallback_recipe(ingredients, dietary_preferences)

async def safe_db_operation(operation, *args, **kwargs):
    """Database safe operation handler (backward compatibility)"""
    return await mongo_manager.safe_operation(operation, *args, **kwargs)

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
    related= []
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
            recipe_copy["match_count"]= match_count
            recipe_copy["match_percentage"]= round(
               (match_count / len(recipe["ingredients"])) * 100)
                related.append(recipe_copy)

                related.sort(key=lambda x: x["match_count"], reverse=True)
                return {"related_recipes": related[:5]}

                    # Updated FastAPI startup event


@ app.on_event("startup")
                    async def startup_db_client():
    """Initialize MongoDB connection on startup"""
    try:
        await mongo_manager.initialize_connection()
        logger.info("Successfully initialized MongoDB connection")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB connection: {e}")
        # Don't raise - let the app start and handle connection issues per request
        pass
                    async def startup():
                        logger.info("Starting Pizza Generator API")
                        logger.info(f"Using model: {os.getenv('LLAMA_MODEL')}")
                        logger.info(f"API base URL: {LLAMA_API_BASE_URL}")

                    # Updated generate_recipe endpoint with proper error handling


@@app.post("/api/generate-recipe", response_model_exclude={"_id"})
async def generate_recipe(request: RecipeRequest):
    recipe_data = jsonable_encoder(recipe)
    try:
        logger.info(
            f"Received generate recipe request: session_id='{request.session_id}' ingredients={request.ingredients} dietary_preferences={request.dietary_preferences} recipe_type='{request.recipe_type}'")

            # Use the mongo manager for all DB operations
            await mongo_manager.safe_operation(
            mongo_manager.sessions_collection.update_one,
            {"session_id": request.session_id},
            {"$set": {
                "session_id": request.session_id,
                "ingredients": request.ingredients,
                "dietary_preferences": request.dietary_preferences.dict(),
                "timestamp": datetime.utcnow()
            }},
            upsert = True
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
                recipe_data.copy()  # Use copy to avoid modifying original
            )
                logger.info(f"Successfully saved recipe to database")
            except Exception as e:
            logger.error(f"Failed to insert recipe into DB: {e}")
            # Continue without raising - recipe generation succeeded even if DB save failed

            recipe_data.pop("_id", None)
            return JSONResponse(
            status_code = 200,
            content ={
                "recipe": recipe_data,
                "status": "success",
                "api_source": recipe_data["api_source"]
            }
        )

        except HTTPException:
        raise
        except Exception as e:
        logger.error(f"Unexpected error in generate-recipe: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@ app.get("/api/recipe/{recipe_id}")
        async def get_recipe(recipe_id: str):
        recipe = await mongo_manager.safe_operation(
        mongo_manager.recipes_collection.find_one,
        {"id": recipe_id}
    )

        if not recipe:
    raise HTTPException(status_code=404, detail="Recipe not found")

        # Remove MongoDB ObjectId from response
        recipe.pop("_id", None)
        return JSONResponse(content={"recipe": recipe})


@ app.get("/api/session/{session_id}/recipes")
    async def get_session_recipes(session_id: str):
    """Get all recipes for a specific session"""
    try:
    recipes_cursor = mongo_manager.recipes_collection.find(
           {"session_id": session_id}
        ).sort("created_at", -1)

            recipes = await mongo_manager.safe_operation(
           recipes_cursor.to_list,
            length = 50  # Limit to 50 recipes per session
        )

            # Remove MongoDB ObjectIds
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
            status_code =500, detail="Failed to fetch session recipes")


@ app.get("/api/recipes/recent")
                async def get_recent_recipes(limit: int = 10):
    """Get recent recipes across all sessions"""
    try:
        if limit > 50:
            limit= 50  # Cap the limit

        recipes_cursor= mongo_manager.recipes_collection.find(
            {},
            {"session_id": 0}  # Exclude session_id for privacy
        ).sort("created_at", -1).limit(limit)

            recipes = await mongo_manager.safe_operation(
           recipes_cursor.to_list,
            length = limit
        )

            # Remove MongoDB ObjectIds
            for recipe in recipes:
        recipe.pop("_id", None)

            return JSONResponse(content={
            "recipes": recipes,
            "count": len(recipes)
        })

        except Exception as e:
        logger.error(f"Error fetching recent recipes: {e}")
        raise HTTPException(
            status_code =500, detail="Failed to fetch recent recipes")


@ app.delete("/api/recipe/{recipe_id}")
            async def delete_recipe(recipe_id: str):
            """Delete a specific recipe"""
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


@ app.post("/api/recipe/{recipe_id}/save")
        async def save_recipe_to_favorites(recipe_id: str, session_id: str):
    """Mark a recipe as favorite for a session"""
        try:
        # Check if recipe exists
        recipe= await mongo_manager.safe_operation(
           mongo_manager.recipes_collection.find_one,
            {"id": recipe_id}
        )

            if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

            # Update recipe to mark as favorite
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


@ app.get("/api/session/{session_id}/favorites")
        async def get_favorite_recipes(session_id: str):
    """Get favorite recipes for a session"""
        try:
        recipes_cursor= mongo_manager.recipes_collection.find(
           {"session_id": session_id, "is_favorite": True}
        ).sort("favorited_at", -1)

            recipes = await mongo_manager.safe_operation(
           recipes_cursor.to_list,
            length = 50
        )

            # Remove MongoDB ObjectIds
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
            status_code =500, detail="Failed to fetch favorite recipes")


@ app.get("/api/stats")
                async def get_api_stats():
    """Get API usage statistics"""
    try:
        # Get total recipes count
        total_recipes= await mongo_manager.safe_operation(
            mongo_manager.recipes_collection.count_documents,
            {}
        )

            # Get total sessions count
            total_sessions = await mongo_manager.safe_operation(
           mongo_manager.sessions_collection.count_documents,
            {}
        )

            # Get recipes created today
            today_start = datetime.now(timezone.utc).replace(
            hour =0, minute=0, second=0, microsecond=0)
            recipes_today = await mongo_manager.safe_operation(
            mongo_manager.recipes_collection.count_documents,
            {"created_at": {"$gte": today_start.isoformat()}}
        )

            # Get most popular ingredients (simplified)
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
            status_code = 500,
            content = {"error": "Failed to fetch statistics"}
        )


@ app.post("/api/feedback")
        async def submit_feedback(feedback_data: dict):
        """Submit user feedback"""
        try:
        feedback_entry= {
           "id": str(uuid.uuid4()),
            "rating": feedback_data.get("rating"),
            "comment": feedback_data.get("comment", ""),
            "recipe_id": feedback_data.get("recipe_id"),
            "session_id": feedback_data.get("session_id"),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "type": "recipe_feedback"
        }

            # Create feedback collection if it doesn't exist
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
            status_code =500, detail="Failed to submit feedback")

            # Cleanup endpoint for development/testing


@ app.delete("/api/cleanup/{session_id}")
            async def cleanup_session(session_id: str):
            """Clean up all data for a specific session (for development/testing)"""
            try:
            # Delete all recipes for the session
            recipes_result = await mongo_manager.safe_operation(
            mongo_manager.recipes_collection.delete_many,
            {"session_id": session_id}
        )

            # Delete session data
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
            status_code =500, detail="Failed to cleanup session")

            # Updated shutdown event


@ app.on_event("shutdown")
            async def shutdown_db_client():
            """Clean up MongoDB connection on shutdown"""
            try:
            if mongo_manager.client:
            mongo_manager.client.close()
            logger.info("MongoDB connection closed")
            except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")

            # Error handling for MongoDB connection issues


@ app.middleware("http")
            async def db_connection_middleware(request: Request, call_next):
            """Middleware to handle database connection issues"""
            try:
            response = await call_next(request)
            return response
            except pymongo.errors.ServerSelectionTimeoutError:
            logger.error("Database connection timeout")
            return JSONResponse(
            status_code = 503,
            content = {"detail": "Database temporarily unavailable"}
        )
        except Exception as e:
        logger.error(f"Unexpected error in middleware: {e}")
        return JSONResponse(
            status_code = 500,
            content = {"detail": "Internal server error"}
        )

        # Run the application
        if __name__ == "__main__":
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run(
        "server:app",
        host = "0.0.0.0",
        port = port,
        reload = False,  # Disable reload in production
        log_level = "info"
    )
