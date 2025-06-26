import requests
import unittest
import json
import uuid
from typing import List, Dict, Any

class PizzaGeneratorAPITest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use the public endpoint from frontend/.env
        self.base_url = "https://f549e02b-fe2f-4efe-9974-2cde4d552c35.preview.emergentagent.com"
        self.session_id = f"test_session_{uuid.uuid4()}"
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "total": 0
        }

    def setUp(self):
        print(f"\n{'='*50}")
        print(f"Running test: {self._testMethodName}")
        print(f"{'-'*50}")
        self.test_results["total"] += 1

    def tearDown(self):
        if hasattr(self, '_outcome'):
            result = self.defaultTestResult()
            self._feedErrorsToResult(result, self._outcome.errors)
            if result.wasSuccessful():
                self.test_results["passed"] += 1
                print(f"✅ Test PASSED: {self._testMethodName}")
            else:
                self.test_results["failed"] += 1
                print(f"❌ Test FAILED: {self._testMethodName}")

    def test_01_health_endpoint(self):
        """Test the health endpoint to verify API is running"""
        response = requests.get(f"{self.base_url}/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "AI Pizza Generator API")
        print(f"Health check response: {data}")

    def test_02_get_ingredients(self):
        """Test fetching all ingredient categories"""
        response = requests.get(f"{self.base_url}/api/ingredients")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("categories", data)
        
        # Verify expected categories exist
        expected_categories = ["flours", "cheeses", "meats", "vegetables", "sauces", "spices_herbs", "other_toppings"]
        for category in expected_categories:
            self.assertIn(category, data["categories"])
            self.assertTrue(len(data["categories"][category]) > 0)
        
        print(f"Found {len(data['categories'])} ingredient categories")

    def test_03_get_ingredients_by_category(self):
        """Test fetching ingredients for a specific category"""
        category = "cheeses"
        response = requests.get(f"{self.base_url}/api/ingredients/{category}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["category"], category)
        self.assertTrue(len(data["ingredients"]) > 0)
        print(f"Found {len(data['ingredients'])} ingredients in category '{category}'")
        
        # Test invalid category
        response = requests.get(f"{self.base_url}/api/ingredients/invalid_category")
        self.assertEqual(response.status_code, 404)

    def test_04_check_conflicts(self):
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
        
        response = requests.post(
            f"{self.base_url}/api/check-conflicts",
            json=payload
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(len(data["conflicts"]) > 0)
        print(f"Found {len(data['conflicts'])} conflicts with vegan diet")
        
        # Test with allergen avoidance
        payload = {
            "ingredients": ["Mozzarella"],
            "dietary_preferences": {
                "diet_types": [],
                "additional_preferences": [],
                "spice_level": 0,
                "allergen_avoidance": ["Dairy"]
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/check-conflicts",
            json=payload
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(len(data["conflicts"]) > 0)
        print(f"Found {len(data['conflicts'])} conflicts with allergen avoidance")

    def test_05_find_related_recipes(self):
        """Test finding related recipes based on ingredients"""
        # Test with ingredients that should match existing recipes
        payload = ["Pepperoni", "Mozzarella", "Classic marinara (V)"]
        
        response = requests.post(
            f"{self.base_url}/api/find-related-recipes",
            json=payload
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("related_recipes", data)
        print(f"Found {len(data['related_recipes'])} related recipes")
        
        if len(data["related_recipes"]) > 0:
            recipe = data["related_recipes"][0]
            self.assertIn("name", recipe)
            self.assertIn("ingredients", recipe)
            self.assertIn("match_count", recipe)
            self.assertIn("match_percentage", recipe)
            print(f"Top match: {recipe['name']} with {recipe['match_count']} matching ingredients ({recipe['match_percentage']}%)")

    def test_06_generate_recipe(self):
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
        
        response = requests.post(
            f"{self.base_url}/api/generate-recipe",
            json=payload
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("recipe", data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "success")
        
        recipe = data["recipe"]
        self.assertIn("id", recipe)
        self.assertIn("name", recipe)
        self.assertIn("ingredients", recipe)
        self.assertIn("steps", recipe)
        self.assertIn("sauce_preparation", recipe)
        self.assertIn("tips", recipe)
        
        print(f"Generated recipe: {recipe['name']}")
        print(f"Recipe ID: {recipe['id']}")
        print(f"Number of steps: {len(recipe['steps'])}")
        
        # Store recipe ID for next test
        self.recipe_id = recipe["id"]
        return recipe["id"]

    def test_07_get_recipe(self):
        """Test retrieving a specific recipe by ID"""
        # First generate a recipe to get an ID
        recipe_id = self.test_06_generate_recipe()
        
        # Now retrieve it
        response = requests.get(f"{self.base_url}/api/recipe/{recipe_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("recipe", data)
        
        recipe = data["recipe"]
        self.assertEqual(recipe["id"], recipe_id)
        print(f"Successfully retrieved recipe: {recipe['name']}")
        
        # Test invalid recipe ID
        response = requests.get(f"{self.base_url}/api/recipe/invalid_id")
        self.assertEqual(response.status_code, 404)

    def test_08_save_cooking_progress(self):
        """Test saving cooking progress"""
        payload = {
            "session_id": self.session_id,
            "step_number": 1,
            "completed": True
        }
        
        response = requests.post(
            f"{self.base_url}/api/save-cooking-progress",
            json=payload
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "progress_saved")
        print("Successfully saved cooking progress")
        
        # Test retrieving cooking progress
        response = requests.get(f"{self.base_url}/api/cooking-progress/{self.session_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("progress", data)
        print(f"Retrieved cooking progress: {data}")

    def print_summary(self):
        """Print a summary of all test results"""
        print("\n" + "="*50)
        print(f"TEST SUMMARY: {self.test_results['passed']}/{self.test_results['total']} tests passed")
        print(f"- Passed: {self.test_results['passed']}")
        print(f"- Failed: {self.test_results['failed']}")
        print("="*50)
        return self.test_results["failed"] == 0

if __name__ == "__main__":
    test_suite = unittest.TestLoader().loadTestsFromTestCase(PizzaGeneratorAPITest)
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # Create an instance to print summary
    test_instance = PizzaGeneratorAPITest()
    test_instance.test_results["passed"] = len(test_result.successes) if hasattr(test_result, 'successes') else test_result.testsRun - len(test_result.failures) - len(test_result.errors)
    test_instance.test_results["failed"] = len(test_result.failures) + len(test_result.errors)
    test_instance.test_results["total"] = test_result.testsRun
    
    success = test_instance.print_summary()
    exit(0 if success else 1)