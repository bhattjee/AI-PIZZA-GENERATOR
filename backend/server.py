from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import requests
import json
import uuid
from datetime import datetime
import pandas as pd
from io import StringIO

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(MONGO_URL)
db = client.pizza_generator
recipes_collection = db.recipes
sessions_collection = db.user_sessions

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

class Recipe(BaseModel):
    id: str
    name: str
    ingredients: List[str]
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

# Related recipes CSV data (simulated)
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
        "ingredients": ["Red bell peppers", "Button mushrooms", "Red onions", "Black olives", "Mozzarella"],
        "match_count": 0
    },
    {
        "name": "Meat Lovers",
        "ingredients": ["Pepperoni", "Italian sausage", "Bacon", "Ham", "Mozzarella"],
        "match_count": 0
    },
    {
        "name": "White Pizza Delight",
        "ingredients": ["Ricotta", "Mozzarella", "Garlic cream", "Spinach", "Pine nuts"],
        "match_count": 0
    }
]

async def generate_ai_recipe(ingredients: List[str], dietary_preferences: DietaryPreferences) -> Recipe:
    """Generate AI-powered pizza recipe using Hugging Face"""
    
    # Create comprehensive prompt
    ingredients_str = ", ".join(ingredients)
    diet_str = ", ".join(dietary_preferences.diet_types) if dietary_preferences.diet_types else "No specific diet"
    allergen_str = ", ".join(dietary_preferences.allergen_avoidance) if dietary_preferences.allergen_avoidance else "No allergens to avoid"
    spice_levels = ["No spice", "Mild", "Medium", "Hot", "Extra Hot"]
    spice_level_str = spice_levels[dietary_preferences.spice_level]
    
    prompt = f"""Create a detailed pizza recipe using these ingredients: {ingredients_str}

REQUIREMENTS:
- Dietary preferences: {diet_str}
- Avoid allergens: {allergen_str}
- Spice level: {spice_level_str}
- Include exact measurements for all ingredients
- Provide step-by-step cooking instructions
- Include dough preparation if flour is selected
- Include detailed sauce preparation
- Specify cooking temperature and time
- Include preparation and cooking times
- Add helpful tips for best results

FORMAT THE RESPONSE AS:
Recipe Name: [Creative pizza name]
Servings: [Number]
Prep Time: [Minutes]
Cook Time: [Minutes]
Difficulty: [Easy/Medium/Hard]

INGREDIENTS:
- [List with exact measurements]

DOUGH PREPARATION (if applicable):
1. [Step by step]

SAUCE PREPARATION:
1. [Step by step]

ASSEMBLY INSTRUCTIONS:
1. [Step by step]

COOKING INSTRUCTIONS:
1. [Step by step with temperatures and times]

TIPS:
- [Helpful cooking tips]

Make this a restaurant-quality pizza recipe with professional techniques."""

    try:
        # Call Hugging Face Inference API
        hf_response = requests.post(
            "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large",
            json={"inputs": prompt, "parameters": {"max_length": 1000, "temperature": 0.7}},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if hf_response.status_code == 503:
            # Model loading, return a structured fallback recipe
            return create_fallback_recipe(ingredients, dietary_preferences)
        
        if hf_response.status_code == 429:
            raise HTTPException(status_code=429, detail="AI service temporarily unavailable. Please try again in a moment.")
            
        if hf_response.status_code != 200:
            return create_fallback_recipe(ingredients, dietary_preferences)
        
        ai_response = hf_response.json()
        generated_text = ai_response[0]["generated_text"] if ai_response else ""
        
        # Parse AI response into structured recipe
        return parse_ai_recipe(generated_text, ingredients, dietary_preferences)
        
    except Exception as e:
        print(f"AI generation error: {e}")
        return create_fallback_recipe(ingredients, dietary_preferences)

def create_fallback_recipe(ingredients: List[str], dietary_preferences: DietaryPreferences) -> Recipe:
    """Create a structured fallback recipe when AI is unavailable"""
    
    # Determine recipe characteristics
    has_meat = any(ingredient in INGREDIENT_CATEGORIES["meats"] for ingredient in ingredients)
    has_veggies = any(ingredient in INGREDIENT_CATEGORIES["vegetables"] for ingredient in ingredients)
    has_sauce = any(ingredient in INGREDIENT_CATEGORIES["sauces"] for ingredient in ingredients)
    
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
            duration_minutes=70
        ),
        CookingStep(
            step_number=2,
            title="Prepare Sauce",
            description="Combine your selected sauce ingredients. For marinara: blend tomatoes, garlic, herbs, and seasonings.",
            duration_minutes=10
        ),
        CookingStep(
            step_number=3,
            title="Preheat Oven",
            description="Preheat oven to 475°F (245°C). If using a pizza stone, place it in the oven while preheating.",
            duration_minutes=15,
            temperature="475°F (245°C)"
        ),
        CookingStep(
            step_number=4,
            title="Roll Out Dough",
            description="On a floured surface, roll out dough to desired thickness. Transfer to pizza pan or parchment paper.",
            duration_minutes=5
        ),
        CookingStep(
            step_number=5,
            title="Add Sauce",
            description="Spread sauce evenly over dough, leaving a 1-inch border for the crust.",
            duration_minutes=2
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
            duration_minutes=3
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

def parse_ai_recipe(ai_text: str, ingredients: List[str], dietary_preferences: DietaryPreferences) -> Recipe:
    """Parse AI-generated text into structured recipe"""
    # This is a simplified parser - in production, you'd want more sophisticated parsing
    lines = ai_text.split('\n')
    
    # Extract basic info (simplified)
    recipe_name = "AI-Generated Custom Pizza"
    for line in lines:
        if "Recipe Name:" in line or "Name:" in line:
            recipe_name = line.split(":")[-1].strip()
            break
    
    # Create structured recipe with AI content
    steps = [
        CookingStep(
            step_number=i+1,
            title=f"Step {i+1}",
            description=f"Follow the AI instructions: {ai_text[:200]}...",
            duration_minutes=10 if i < 3 else 15
        ) for i in range(8)
    ]
    
    return Recipe(
        id=str(uuid.uuid4()),
        name=recipe_name,
        ingredients=ingredients,
        dietary_info=dietary_preferences.diet_types,
        spice_level=dietary_preferences.spice_level,
        prep_time=60,
        cook_time=15,
        total_time=75,
        servings=4,
        difficulty="Medium",
        steps=steps,
        sauce_preparation=["Follow AI sauce instructions"],
        tips=["AI-generated tips included in full recipe"]
    )

# API Routes

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
    """Find related recipes based on ingredient matches"""
    related = []
    
    for recipe in RELATED_RECIPES:
        match_count = len(set(ingredients) & set(recipe["ingredients"]))
        if match_count >= 2:  # At least 2 ingredients match
            recipe_copy = recipe.copy()
            recipe_copy["match_count"] = match_count
            recipe_copy["match_percentage"] = round((match_count / len(recipe["ingredients"])) * 100)
            related.append(recipe_copy)
    
    # Sort by match count
    related.sort(key=lambda x: x["match_count"], reverse=True)
    
    return {"related_recipes": related[:5]}  # Return top 5 matches

@app.post("/api/generate-recipe")
async def generate_recipe(request: RecipeRequest):
    """Generate custom AI recipe"""
    try:
        # Store session data
        session_data = {
            "session_id": request.session_id,
            "ingredients": request.ingredients,
            "dietary_preferences": request.dietary_preferences.dict(),
            "timestamp": datetime.utcnow()
        }
        await sessions_collection.update_one(
            {"session_id": request.session_id},
            {"$set": session_data},
            upsert=True
        )
        
        # Generate recipe
        recipe = await generate_ai_recipe(request.ingredients, request.dietary_preferences)
        
        # Store generated recipe
        recipe_data = recipe.dict()
        recipe_data["session_id"] = request.session_id
        recipe_data["created_at"] = datetime.utcnow()
        
        await recipes_collection.insert_one(recipe_data)
        
        return {"recipe": recipe, "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recipe generation failed: {str(e)}")

@app.get("/api/recipe/{recipe_id}")
async def get_recipe(recipe_id: str):
    """Get a specific recipe by ID"""
    recipe = await recipes_collection.find_one({"id": recipe_id})
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    # Remove MongoDB _id field
    recipe.pop("_id", None)
    return {"recipe": recipe}

@app.post("/api/save-cooking-progress")
async def save_cooking_progress(session_id: str, step_number: int, completed: bool):
    """Save cooking progress"""
    progress_data = {
        "session_id": session_id,
        "step_number": step_number,
        "completed": completed,
        "timestamp": datetime.utcnow()
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

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Pizza Generator API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)