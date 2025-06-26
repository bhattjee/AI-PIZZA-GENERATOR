import requests
import sys
import uuid
import json

class PizzaGeneratorAPITester:
    def __init__(self):
        # Use the public endpoint from frontend/.env
        self.base_url = "https://f549e02b-fe2f-4efe-9974-2cde4d552c35.preview.emergentagent.com"
        self.session_id = f"test_session_{uuid.uuid4()}"
        self.tests_run = 0
        self.tests_passed = 0
        self.recipe_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, params=None):
        """Run a single API test"""
        url = f"{self.base_url}/api/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method == 'POST':
                if params:
                    # If params are provided, send as query parameters
                    response = requests.post(url, headers=headers, params=params)
                else:
                    # Otherwise send as JSON body
                    response = requests.post(url, json=data, headers=headers)
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    return success, response.json()
                except:
                    return success, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    print(f"Response: {response.text}")
                except:
                    pass
                return False, {}
                
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_health_endpoint(self):
        """Test the health endpoint to verify API is running"""
        success, data = self.run_test(
            "Health Endpoint",
            "GET",
            "health",
            200
        )
        if success:
            print(f"Health check response: {data}")
        return success

    def test_get_ingredients(self):
        """Test fetching all ingredient categories"""
        success, data = self.run_test(
            "Get All Ingredients",
            "GET",
            "ingredients",
            200
        )
        if success and "categories" in data:
            categories = data["categories"]
            print(f"Found {len(categories)} ingredient categories")
            for category, ingredients in categories.items():
                print(f"- {category}: {len(ingredients)} ingredients")
        return success

    def test_get_ingredients_by_category(self):
        """Test fetching ingredients for a specific category"""
        category = "cheeses"
        success, data = self.run_test(
            f"Get Ingredients by Category ({category})",
            "GET",
            f"ingredients/{category}",
            200
        )
        if success:
            print(f"Found {len(data['ingredients'])} ingredients in category '{category}'")
        
        # Test invalid category
        invalid_success, _ = self.run_test(
            "Get Ingredients with Invalid Category",
            "GET",
            "ingredients/invalid_category",
            404
        )
        return success and invalid_success

    def test_check_conflicts(self):
        """Test checking conflicts between ingredients and dietary preferences"""
        # Test with vegan diet and non-vegan ingredients
        payload = {
            "ingredients": ["Pepperoni", "Mozzarella"],
            "dietary_preferences": {
                "diet_types": ["Vegan"],
                "additional_preferences": [],
                "spice_level": 0,
                "allergen_avoidance": []
            }
        }
        
        success, data = self.run_test(
            "Check Conflicts (Vegan Diet)",
            "POST",
            "check-conflicts",
            200,
            data=payload
        )
        
        if success:
            conflicts = data.get("conflicts", [])
            print(f"Found {len(conflicts)} conflicts with vegan diet")
            for conflict in conflicts:
                print(f"- {conflict['ingredient']}: {conflict['conflict_detail']}")
        
        return success

    def test_find_related_recipes(self):
        """Test finding related recipes based on ingredients"""
        payload = ["Pepperoni", "Mozzarella", "Classic marinara (V)"]
        
        success, data = self.run_test(
            "Find Related Recipes",
            "POST",
            "find-related-recipes",
            200,
            data=payload
        )
        
        if success and "related_recipes" in data:
            recipes = data["related_recipes"]
            print(f"Found {len(recipes)} related recipes")
            for recipe in recipes:
                print(f"- {recipe['name']} ({recipe['match_percentage']}% match)")
        
        return success

    def test_generate_recipe(self):
        """Test generating a custom AI recipe"""
        payload = {
            "session_id": self.session_id,
            "ingredients": ["All-purpose", "Mozzarella", "Tomatoes", "Classic marinara (V)", "Oregano"],
            "dietary_preferences": {
                "diet_types": [],
                "additional_preferences": [],
                "spice_level": 1,
                "allergen_avoidance": []
            },
            "recipe_type": "custom"
        }
        
        success, data = self.run_test(
            "Generate Custom Recipe",
            "POST",
            "generate-recipe",
            200,
            data=payload
        )
        
        if success and "recipe" in data:
            recipe = data["recipe"]
            print(f"Generated recipe: {recipe['name']}")
            print(f"Recipe ID: {recipe['id']}")
            print(f"Number of steps: {len(recipe['steps'])}")
            self.recipe_id = recipe["id"]
        
        return success

    def test_get_recipe(self):
        """Test retrieving a specific recipe by ID"""
        if not self.recipe_id:
            print("❌ Cannot test get_recipe without a valid recipe ID")
            return False
        
        success, data = self.run_test(
            "Get Recipe by ID",
            "GET",
            f"recipe/{self.recipe_id}",
            200
        )
        
        if success and "recipe" in data:
            recipe = data["recipe"]
            print(f"Retrieved recipe: {recipe['name']}")
        
        # Test invalid recipe ID
        invalid_success, _ = self.run_test(
            "Get Recipe with Invalid ID",
            "GET",
            "recipe/invalid_id",
            404
        )
        
        return success and invalid_success

    def test_save_cooking_progress(self):
        """Test saving cooking progress"""
        # The API expects query parameters, not JSON body
        params = {
            "session_id": self.session_id,
            "step_number": 1,
            "completed": True
        }
        
        success, data = self.run_test(
            "Save Cooking Progress",
            "POST",
            "save-cooking-progress",
            200,
            params=params
        )
        
        if success:
            print("Successfully saved cooking progress")
        
        # Test retrieving cooking progress
        get_success, progress_data = self.run_test(
            "Get Cooking Progress",
            "GET",
            f"cooking-progress/{self.session_id}",
            200
        )
        
        if get_success:
            print(f"Retrieved cooking progress: {progress_data}")
        
        return success and get_success

    def run_all_tests(self):
        """Run all API tests"""
        print("="*50)
        print("RUNNING API TESTS FOR AI PIZZA GENERATOR")
        print("="*50)
        
        tests = [
            self.test_health_endpoint,
            self.test_get_ingredients,
            self.test_get_ingredients_by_category,
            self.test_check_conflicts,
            self.test_find_related_recipes,
            self.test_generate_recipe,
            self.test_get_recipe,
            self.test_save_cooking_progress
        ]
        
        for test in tests:
            test()
        
        print("\n" + "="*50)
        print(f"TEST SUMMARY: {self.tests_passed}/{self.tests_run} tests passed")
        print("="*50)
        
        return self.tests_passed == self.tests_run

if __name__ == "__main__":
    tester = PizzaGeneratorAPITester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)