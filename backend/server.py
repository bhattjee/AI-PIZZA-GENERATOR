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
from datetime import datetime
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
import logging
from fastapi.responses import JSONResponse
from datetime import timezone

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
        "https://ai-pizza-frontend.vercel.app",
        "https://pizzacrust.onrender.com"
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add OPTIONS handler

@app.options("/{path:path}")
async def options_handler():
    return {"message": "OK"}

# Validate environment variables
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY")
if not LLAMA_API_KEY:
    logger.error("LLAMA_API_KEY not found in environment variables")
    raise ValueError("LLAMA_API_KEY is required")

# MongoDB URL (defaulting to localhost)
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")

# Correct Hugging Face inference API base URL for Mistral
LLAMA_API_BASE_URL = os.environ.get(
    "LLAMA_API_BASE_URL",
    "https://openrouter.ai/api/v1/chat/completions"
)


# Request timeout setting
LLAMA_TIMEOUT = 30  # seconds

# Database connection
try:
    client = AsyncIOMotorClient(MONGO_URL)
    db = client.pizza_generator
    recipes_collection = db.recipes
    sessions_collection = db.user_sessions
    logger.info("MongoDB connection established")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

# Global exception handler


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"},
    )

# Health check endpoint


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
        headers={
            "Authorization": f"Bearer {LLAMA_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",  # or your deployed domain
            "X-Title": "ai-pizza-generator"
        }
    )


@app.get("/health")
async def health_check():
    try:
        await client.server_info()  # Test DB connection
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)},
        )

# Handle OPTIONS requests for CORS preflight


@app.options("/{path:path}")
async def options_handler():
    return {"message": "OK"}

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
    """Generate recipe using Hugging Face Inference API with Mistral-7B"""
    try:
        if not LLAMA_API_KEY:
            logger.error("LLAMA_API_KEY not configured")
            return None

        # Enhanced prompt with strict JSON formatting instructions
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
      "duration_minutes": 10,  // Must not be null or missing
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
    }},
    {{
      "step_number": 2,
      ...
    }}
  ],
  "tips": ["Helpful tip 1", "Helpful tip 2"],
  "nutrition": {{
    "calories": 500,
    "protein": 20,
    "carbs": 60,
    "fat": 15
  }}
}}

Important:
1. Return ONLY valid JSON (no commentary or markdown)
2. All fields must be included
3. Every step (main or sauce) MUST have duration_minutes and a proper list of used ingredients
4. Keep measurements consistent (minutes for time, grams/oz for weights)
5. Make the recipe creative but practical and detailed
"""

        payload = {
            "model": "mistralai/mistral-7b-instruct",  # or "meta-llama/llama-3-8b-instruct"
            "messages": [
                {"role": "system", "content": "You are a helpful pizza recipe assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1500
        }

        logger.info(
            f"Sending request to Mistral-7B API with payload: {payload}")

        headers = {
            "Authorization": f"Bearer {LLAMA_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "ai-pizza-generator"
        }

        response = requests.post(
            LLAMA_API_BASE_URL,  # Now points to Mistral endpoint
            headers=headers,
            json=payload,
            timeout=LLAMA_TIMEOUT
        )

        logger.info(f"API response status: {response.status_code}")

        if response.status_code == 200:
            try:
                # Handle potential JSON wrapping in response
                response_text = response.json(
                )["choices"][0]["message"]["content"]

                # Clean the response (remove markdown formatting if present)
                clean_json = response_text.replace(
                    '```json', '').replace('```', '').strip()

                # Parse and validate the JSON
                recipe_data = json.loads(clean_json)
                logger.info("Successfully parsed recipe from API response")
                return recipe_data

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse API response: {str(e)}")
                # Log first 200 chars
                logger.error(f"Raw response: {response_text[:200]}...")
                return None

        else:
            logger.error(f"API error {response.status_code}: {response.text}")
            return None

    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in API call: {str(e)}", exc_info=True)
        return None


def convert_llama_response_to_recipe(llama_response: Dict, ingredients: List[str], dietary_preferences: DietaryPreferences) -> Recipe:
    """Convert LLAMA API response to our Recipe model"""

    if not llama_response:
        return create_fallback_recipe(ingredients, dietary_preferences)

    # Extract steps
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

    # Extract nutrition
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


async def generate_llama_recipe(ingredients: List[str], dietary_preferences: DietaryPreferences) -> Recipe:
    """Generate recipe using LLAMA API with fallback"""
    try:
        # First try with LLAMA API
        llama_response = await generate_recipe_with_llama(ingredients, dietary_preferences)

        if llama_response:
            return convert_llama_response_to_recipe(llama_response, ingredients, dietary_preferences)
        else:
            # Fallback if API fails
            logger.warning("LLAMA API failed, using fallback recipe")
            return create_fallback_recipe(ingredients, dietary_preferences)

    except Exception as e:
        logger.error(f"Recipe generation failed: {e}")
        return create_fallback_recipe(ingredients, dietary_preferences)


def create_fallback_recipe(ingredients: List[str], dietary_preferences: DietaryPreferences) -> Recipe:
    """Create a structured fallback recipe when API is unavailable"""

    # Determine recipe characteristics
    has_meat = any(
        ingredient in INGREDIENT_CATEGORIES["meats"] for ingredient in ingredients)
    has_veggies = any(
        ingredient in INGREDIENT_CATEGORIES["vegetables"] for ingredient in ingredients)
    has_sauce = any(
        ingredient in INGREDIENT_CATEGORIES["sauces"] for ingredient in ingredients)

    # Generate recipe name
    if has_meat and has_veggies:
        recipe_name = "Hearty Meat & Veggie Pizza"
    elif has_meat:
        recipe_name = "Savory Meat Lovers Pizza"
    elif has_veggies:
        recipe_name = "Garden Fresh Veggie Pizza"
    else:
        recipe_name = "Custom Artisan Pizza"

    # Create cooking steps
    steps = [
        CookingStep(
            step_number=1,
            title="Prepare the Dough",
            description="If using flour, mix with warm water, yeast, salt, and olive oil. Knead for 8-10 minutes until smooth. Let rise for 1 hour.",
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

# API Routes


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Pizza Generator API", "status": "running"}


@app.get("/api/ingredients")
async def get_ingredients():
    """Get all ingredient categories"""
    return {"categories": INGREDIENT_CATEGORIES}


@app.get("/api/ingredients/{category}")
async def get_ingredients_by_category(category: str):
    """Get ingredients for a specific category"""
    if category not in INGREDIENT_CATEGORIES:
        raise HTTPException(status_code=404, detail="Category not found")
    return {"category": category, "ingredients": INGREDIENT_CATEGORIES[category]}


@app.post("/api/check-conflicts")
async def check_conflicts(ingredients: List[str], dietary_preferences: DietaryPreferences):
    """Check for conflicts between ingredients and dietary preferences"""
    conflicts = []

    # Check allergen conflicts
    for ingredient in ingredients:
        for allergen in dietary_preferences.allergen_avoidance:
            if allergen.lower() in ingredient.lower():
                conflicts.append({
                    "ingredient": ingredient,
                    "conflict_type": "allergen",
                    "conflict_detail": f"Contains {allergen}"
                })

    # Check diet conflicts
    if "vegan" in [d.lower() for d in dietary_preferences.diet_types]:
        non_vegan_ingredients = []
        for ingredient in ingredients:
            if any(meat in ingredient.lower() for meat in ["cheese", "meat", "bacon", "ham", "pepperoni"]):
                non_vegan_ingredients.append(ingredient)
        for ingredient in non_vegan_ingredients:
            conflicts.append({
                "ingredient": ingredient,
                "conflict_type": "diet",
                "conflict_detail": "Not vegan"
            })

    return {"conflicts": conflicts}


@app.post("/api/find-related-recipes")
async def find_related_recipes(ingredients: List[str]):
    """Find related recipes based on selected ingredients"""
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


@app.post("/api/generate-recipe")
async def generate_recipe(request: RecipeRequest):
    """Generate custom recipe using LLAMA API"""
    try:
        logger.info(f"Received generate recipe request: {request}")

        # 🧠 Ensure dietary_preferences are serializable
        if hasattr(request.dietary_preferences, "model_dump"):
            dietary_preferences_data = request.dietary_preferences.model_dump()
        else:
            dietary_preferences_data = request.dietary_preferences

        session_data = {
            "session_id": request.session_id,
            "ingredients": request.ingredients,
            "dietary_preferences": dietary_preferences_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        await sessions_collection.update_one(
            {"session_id": request.session_id},
            {"$set": session_data},
            upsert=True
        )

        # 🧠 Generate recipe (LLAMA or fallback)
        recipe = await generate_llama_recipe(
            request.ingredients,
            request.dietary_preferences
        )

        # ✅ MAIN FIX: jsonable_encoder handles model, ObjectId, etc.
        recipe_data = jsonable_encoder(recipe)

        # Attach metadata
        recipe_data["session_id"] = request.session_id
        recipe_data["created_at"] = datetime.now(timezone.utc).isoformat()
        recipe_data["api_source"] = "llama"

        try:
            await recipes_collection.insert_one(recipe_data)
        except Exception as e:
            logger.error(f"Failed to insert recipe into DB: {e}")

        # Remove MongoDB _id field if exists
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
        logger.error(
            f"Unexpected error in generate-recipe: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.get("/api/recipe/{recipe_id}")
async def get_recipe(recipe_id: str):
    """Get a specific recipe by ID"""
    recipe = await recipes_collection.find_one({"id": recipe_id})
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    # Remove MongoDB _id field
    recipe.pop("_id", None)
    return {"recipe": recipe}


@app.get("/api/recipes/session/{session_id}")
async def get_session_recipes(session_id: str):
    """Get all recipes for a session"""
    recipes = []
    async for recipe in recipes_collection.find({"session_id": session_id}):
        recipe.pop("_id", None)
        recipes.append(recipe)

    return {"recipes": recipes}


@app.post("/api/save-cooking-progress")
async def save_cooking_progress(session_id: str, step_number: int, completed: bool):
    """Save cooking progress"""
    progress_data = {
        "session_id": session_id,
        "step_number": step_number,
        "completed": completed,
        "timestamp": datetime.now(timezone.utc).isoformat()  # 👈 FIXED
    }

    await sessions_collection.update_one(
        {"session_id": session_id},
        {"$set": {f"cooking_progress.step_{step_number}": progress_data}},
        upsert=True
    )

    return {"status": "progress_saved"}


@app.get("/api/cooking-progress/{session_id}")
async def get_cooking_progress(session_id: str):
    """Get cooking progress for a session"""
    session = await sessions_collection.find_one({"session_id": session_id})
    if not session:
        return {"progress": {}}

    return {"progress": session.get("cooking_progress", {})}


@app.post("/api/remove-ingredient")
async def remove_ingredient(session_id: str, ingredient: str):
    """Remove an ingredient from the current session"""
    try:
        # Get current session
        session = await sessions_collection.find_one({"session_id": session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Remove ingredient if it exists
        current_ingredients = session.get("ingredients", [])
        if ingredient in current_ingredients:
            updated_ingredients = [
                i for i in current_ingredients if i != ingredient]

            # Update session
            await sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": {"ingredients": updated_ingredients}}
            )

            return {"status": "removed", "remaining_ingredients": updated_ingredients}
        else:
            return {"status": "not_found", "message": "Ingredient not in current selection"}

    except Exception as e:
        logger.error(f"Failed to remove ingredient: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to remove ingredient: {str(e)}"
        )


@app.post("/api/validate-recipe")
async def validate_recipe(recipe: Recipe):
    """Validate recipe structure and completeness"""
    errors = []

    # Check required fields
    if not recipe.name:
        errors.append("Recipe name is required")
    if not recipe.ingredients:
        errors.append("At least one ingredient is required")
    if not recipe.steps:
        errors.append("At least one cooking step is required")

    # Check step completeness
    for i, step in enumerate(recipe.steps):
        if not step.description:
            errors.append(f"Step {i+1} is missing a description")

    # Check nutrition if provided
    if recipe.nutrition:
        if recipe.nutrition.calories is not None and recipe.nutrition.calories < 0:
            errors.append("Calories cannot be negative")
        if recipe.nutrition.protein is not None and recipe.nutrition.protein < 0:
            errors.append("Protein cannot be negative")

    if errors:
        return {
            "valid": False,
            "errors": errors
        }
    else:
        return {
            "valid": True,
            "message": "Recipe is valid"
        }

# Error handlers


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Consistent port
