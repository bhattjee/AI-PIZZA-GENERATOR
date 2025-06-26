import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

const STEPS = {
  INGREDIENTS: 1,
  PREFERENCES: 2,
  RELATED: 3,
  GENERATE: 4,
  COOKING: 5
};

const SPICE_LEVELS = ['No Spice', 'Mild 🌶', 'Medium 🌶🌶', 'Hot 🌶🌶🌶', 'Extra Hot 🌶🌶🌶🌶'];

const DIET_OPTIONS = [
  'Vegan', 'Vegetarian', 'Gluten-Free', 'Keto', 'Paleo', 'Low-Carb', 
  'Dairy-Free', 'Nut-Free', 'Egg-Free', 'Soy-Free', 'Low-Fat', 
  'High-Protein', 'Diabetic-Friendly', 'Heart-Healthy', 'Mediterranean'
];

const ADDITIONAL_PREFERENCES = [
  'Organic', 'Non-GMO', 'Local', 'Seasonal', 'Sustainable', 
  'Kosher', 'Low-Sodium', 'Sugar-Free', 'High-Fiber'
];

const ALLERGENS = [
  'Dairy', 'Nuts', 'Gluten', 'Eggs', 'Soy', 'Shellfish', 
  'Fish', 'Sesame', 'Mustard', 'Sulfites'
];

function App() {
  const [currentStep, setCurrentStep] = useState(STEPS.INGREDIENTS);
  const [darkMode, setDarkMode] = useState(false);
  const [sessionId] = useState(() => 'session_' + Date.now());
  
  // State for ingredient selection
  const [ingredientCategories, setIngredientCategories] = useState({});
  const [selectedCategory, setSelectedCategory] = useState('flours');
  const [selectedIngredients, setSelectedIngredients] = useState([]);
  
  // State for dietary preferences
  const [dietaryPreferences, setDietaryPreferences] = useState({
    diet_types: [],
    additional_preferences: [],
    spice_level: 0,
    allergen_avoidance: []
  });
  
  // State for conflicts
  const [conflicts, setConflicts] = useState([]);
  const [showConflictModal, setShowConflictModal] = useState(false);
  
  // State for recipes
  const [relatedRecipes, setRelatedRecipes] = useState([]);
  const [currentRecipe, setCurrentRecipe] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  
  // State for cooking
  const [currentCookingStep, setCurrentCookingStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState(new Set());
  const [timerActive, setTimerActive] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(0);

  useEffect(() => {
    fetchIngredients();
  }, []);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // Timer effect
  useEffect(() => {
    let interval = null;
    if (timerActive && timeRemaining > 0) {
      interval = setInterval(() => {
        setTimeRemaining(time => time - 1);
      }, 1000);
    } else if (timeRemaining === 0) {
      setTimerActive(false);
    }
    return () => clearInterval(interval);
  }, [timerActive, timeRemaining]);

  const fetchIngredients = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/ingredients`);
      const data = await response.json();
      setIngredientCategories(data.categories);
    } catch (error) {
      console.error('Error fetching ingredients:', error);
    }
  };

  const handleIngredientToggle = (ingredient) => {
    setSelectedIngredients(prev => 
      prev.includes(ingredient) 
        ? prev.filter(i => i !== ingredient)
        : [...prev, ingredient]
    );
  };

  const handleDietaryChange = (field, value) => {
    setDietaryPreferences(prev => ({
      ...prev,
      [field]: Array.isArray(prev[field]) 
        ? prev[field].includes(value)
          ? prev[field].filter(item => item !== value)
          : [...prev[field], value]
        : value
    }));
  };

  const checkConflicts = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/check-conflicts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ingredients: selectedIngredients,
          dietary_preferences: dietaryPreferences
        })
      });
      const data = await response.json();
      
      if (data.conflicts.length > 0) {
        setConflicts(data.conflicts);
        setShowConflictModal(true);
        return false;
      }
      return true;
    } catch (error) {
      console.error('Error checking conflicts:', error);
      return true;
    }
  };

  const findRelatedRecipes = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/find-related-recipes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(selectedIngredients)
      });
      const data = await response.json();
      setRelatedRecipes(data.related_recipes);
      
      if (data.related_recipes.length > 0) {
        setCurrentStep(STEPS.RELATED);
      } else {
        generateCustomRecipe();
      }
    } catch (error) {
      console.error('Error finding related recipes:', error);
      generateCustomRecipe();
    }
  };

  const generateCustomRecipe = async () => {
    setIsGenerating(true);
    setCurrentStep(STEPS.GENERATE);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-recipe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          ingredients: selectedIngredients,
          dietary_preferences: dietaryPreferences,
          recipe_type: 'custom'
        })
      });
      
      const data = await response.json();
      setCurrentRecipe(data.recipe);
      setCurrentStep(STEPS.COOKING);
    } catch (error) {
      console.error('Error generating recipe:', error);
      // Handle error appropriately
    } finally {
      setIsGenerating(false);
    }
  };

  const handleStepComplete = async (stepNumber) => {
    const newCompleted = new Set(completedSteps);
    newCompleted.add(stepNumber);
    setCompletedSteps(newCompleted);
    
    // Save progress to backend
    try {
      await fetch(`${API_BASE_URL}/api/save-cooking-progress`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          step_number: stepNumber,
          completed: true
        })
      });
    } catch (error) {
      console.error('Error saving progress:', error);
    }
  };

  const startTimer = (minutes) => {
    setTimeRemaining(minutes * 60);
    setTimerActive(true);
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const renderStepProgress = () => {
    return (
      <div className="step-progress">
        {Object.values(STEPS).map(step => (
          <div
            key={step}
            className={`step-indicator ${currentStep >= step ? 'active' : ''} ${currentStep === step ? 'current' : ''}`}
          >
            {step}
          </div>
        ))}
      </div>
    );
  };

  const renderIngredientSelection = () => {
    return (
      <div className="ingredient-selection">
        <h2>🍕 Choose Your Ingredients</h2>
        <div className="selection-layout">
          <div className="categories-sidebar">
            <h3>Categories</h3>
            {Object.keys(ingredientCategories).map(category => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`category-button ${selectedCategory === category ? 'active' : ''}`}
              >
                {category.charAt(0).toUpperCase() + category.slice(1).replace('_', ' & ')}
              </button>
            ))}
          </div>
          
          <div className="ingredients-grid">
            <h3>{selectedCategory.charAt(0).toUpperCase() + selectedCategory.slice(1).replace('_', ' & ')}</h3>
            <div className="ingredient-buttons">
              {ingredientCategories[selectedCategory]?.map(ingredient => (
                <button
                  key={ingredient}
                  onClick={() => handleIngredientToggle(ingredient)}
                  className={`ingredient-button ${selectedIngredients.includes(ingredient) ? 'selected' : ''}`}
                >
                  {ingredient}
                </button>
              ))}
            </div>
          </div>
        </div>
        
        <div className="selected-ingredients">
          <h3>Selected Ingredients ({selectedIngredients.length})</h3>
          <div className="selected-tags">
            {selectedIngredients.map(ingredient => (
              <span key={ingredient} className="ingredient-tag">
                {ingredient}
                <button onClick={() => handleIngredientToggle(ingredient)}>×</button>
              </span>
            ))}
          </div>
        </div>
        
        <button
          onClick={async () => {
            if (selectedIngredients.length === 0) {
              alert('Please select at least one ingredient');
              return;
            }
            setCurrentStep(STEPS.PREFERENCES);
          }}
          className="next-button"
          disabled={selectedIngredients.length === 0}
        >
          Next: Set Preferences →
        </button>
      </div>
    );
  };

  const renderDietaryPreferences = () => {
    return (
      <div className="dietary-preferences">
        <h2>⚙️ Set Your Dietary Preferences</h2>
        
        <div className="preference-section">
          <h3>Diet Types</h3>
          <div className="preference-grid">
            {DIET_OPTIONS.map(diet => (
              <button
                key={diet}
                onClick={() => handleDietaryChange('diet_types', diet)}
                className={`preference-button ${dietaryPreferences.diet_types.includes(diet) ? 'selected' : ''}`}
              >
                {diet}
              </button>
            ))}
          </div>
        </div>

        <div className="preference-section">
          <h3>Additional Preferences</h3>
          <div className="preference-grid">
            {ADDITIONAL_PREFERENCES.map(pref => (
              <button
                key={pref}
                onClick={() => handleDietaryChange('additional_preferences', pref)}
                className={`preference-button ${dietaryPreferences.additional_preferences.includes(pref) ? 'selected' : ''}`}
              >
                {pref}
              </button>
            ))}
          </div>
        </div>

        <div className="preference-section">
          <h3>Spice Level</h3>
          <div className="spice-slider">
            <input
              type="range"
              min="0"
              max="4"
              value={dietaryPreferences.spice_level}
              onChange={(e) => handleDietaryChange('spice_level', parseInt(e.target.value))}
              className="slider"
            />
            <div className="spice-label">{SPICE_LEVELS[dietaryPreferences.spice_level]}</div>
          </div>
        </div>

        <div className="preference-section">
          <h3>Allergen Avoidance</h3>
          <div className="preference-grid">
            {ALLERGENS.map(allergen => (
              <button
                key={allergen}
                onClick={() => handleDietaryChange('allergen_avoidance', allergen)}
                className={`preference-button allergen ${dietaryPreferences.allergen_avoidance.includes(allergen) ? 'selected' : ''}`}
              >
                {allergen}
              </button>
            ))}
          </div>
        </div>

        <div className="navigation-buttons">
          <button onClick={() => setCurrentStep(STEPS.INGREDIENTS)} className="back-button">
            ← Back
          </button>
          <button
            onClick={async () => {
              const hasConflicts = await checkConflicts();
              if (hasConflicts) {
                findRelatedRecipes();
              }
            }}
            className="next-button"
          >
            Next: Find Recipes →
          </button>
        </div>
      </div>
    );
  };

  const renderConflictModal = () => {
    if (!showConflictModal) return null;

    return (
      <div className="modal-overlay">
        <div className="conflict-modal">
          <h3>⚠️ Dietary Conflicts Detected</h3>
          <div className="conflicts-list">
            {conflicts.map((conflict, index) => (
              <div key={index} className="conflict-item">
                <strong>{conflict.ingredient}</strong>: {conflict.conflict_detail}
                <div className="conflict-actions">
                  <button
                    onClick={() => {
                      setSelectedIngredients(prev => prev.filter(i => i !== conflict.ingredient));
                    }}
                    className="remove-ingredient"
                  >
                    Remove Ingredient
                  </button>
                  <button
                    onClick={() => {
                      // Keep ingredient, ignore conflict for this item
                      setConflicts(prev => prev.filter((_, i) => i !== index));
                    }}
                    className="proceed-anyway"
                  >
                    Proceed Anyway
                  </button>
                </div>
              </div>
            ))}
          </div>
          <button
            onClick={() => {
              setShowConflictModal(false);
              setConflicts([]);
              findRelatedRecipes();
            }}
            className="continue-button"
            disabled={conflicts.length > 0}
          >
            Continue to Recipes
          </button>
        </div>
      </div>
    );
  };

  const renderRelatedRecipes = () => {
    return (
      <div className="related-recipes">
        <h2>🔍 Related Recipes Found</h2>
        <p>We found {relatedRecipes.length} recipes that match your ingredients!</p>
        
        <div className="recipes-grid">
          {relatedRecipes.map((recipe, index) => (
            <div key={index} className="recipe-card">
              <h3>{recipe.name}</h3>
              <div className="match-info">
                <span className="match-count">{recipe.match_count} ingredients match</span>
                <span className="match-percentage">{recipe.match_percentage}% match</span>
              </div>
              <div className="recipe-ingredients">
                <h4>Ingredients:</h4>
                <ul>
                  {recipe.ingredients.map(ingredient => (
                    <li key={ingredient} className={selectedIngredients.includes(ingredient) ? 'matched' : ''}>
                      {ingredient}
                    </li>
                  ))}
                </ul>
              </div>
              <button
                onClick={() => {
                  // For now, use the selected recipe to generate custom instructions
                  generateCustomRecipe();
                }}
                className="select-recipe-button"
              >
                Use This Recipe
              </button>
            </div>
          ))}
        </div>
        
        <div className="navigation-buttons">
          <button onClick={() => setCurrentStep(STEPS.PREFERENCES)} className="back-button">
            ← Back
          </button>
          <button onClick={generateCustomRecipe} className="generate-custom-button">
            Generate Custom Recipe Instead →
          </button>
        </div>
      </div>
    );
  };

  const renderRecipeGeneration = () => {
    return (
      <div className="recipe-generation">
        <h2>🤖 Generating Your Custom Recipe</h2>
        <div className="loading-animation">
          <div className="pizza-spinner">🍕</div>
          <p>Our AI chef is crafting the perfect recipe for you...</p>
          <div className="progress-bar">
            <div className="progress-fill"></div>
          </div>
        </div>
        <p>This may take a moment as we analyze your ingredients and preferences.</p>
      </div>
    );
  };

  const renderCookingAssistant = () => {
    if (!currentRecipe) return null;

    const currentStepData = currentRecipe.steps[currentCookingStep];
    const isLastStep = currentCookingStep === currentRecipe.steps.length - 1;
    const allStepsCompleted = completedSteps.size === currentRecipe.steps.length;

    return (
      <div className="cooking-assistant">
        <h2>👨‍🍳 Cooking Assistant</h2>
        
        <div className="recipe-header">
          <h3>{currentRecipe.name}</h3>
          <div className="recipe-meta">
            <span>⏱️ {currentRecipe.total_time} min total</span>
            <span>🍽️ Serves {currentRecipe.servings}</span>
            <span>📊 {currentRecipe.difficulty}</span>
          </div>
        </div>

        {!allStepsCompleted && (
          <div className="current-step">
            <div className="step-header">
              <h4>Step {currentStepData.step_number}: {currentStepData.title}</h4>
              {currentStepData.duration_minutes && (
                <div className="timer-section">
                  <button
                    onClick={() => startTimer(currentStepData.duration_minutes)}
                    className="start-timer-button"
                    disabled={timerActive}
                  >
                    Start {currentStepData.duration_minutes}min Timer
                  </button>
                  {timerActive && (
                    <div className="timer-display">
                      Time Remaining: {formatTime(timeRemaining)}
                    </div>
                  )}
                </div>
              )}
            </div>
            
            <p className="step-description">{currentStepData.description}</p>
            
            {currentStepData.temperature && (
              <div className="temperature-info">
                🌡️ Temperature: {currentStepData.temperature}
              </div>
            )}
            
            <div className="step-actions">
              <button
                onClick={() => {
                  handleStepComplete(currentStepData.step_number);
                  if (!isLastStep) {
                    setCurrentCookingStep(prev => prev + 1);
                  }
                }}
                className="complete-step-button"
                disabled={completedSteps.has(currentStepData.step_number)}
              >
                {completedSteps.has(currentStepData.step_number) ? '✅ Completed' : 'Mark as Complete'}
              </button>
              
              {currentCookingStep > 0 && (
                <button
                  onClick={() => setCurrentCookingStep(prev => prev - 1)}
                  className="previous-step-button"
                >
                  ← Previous Step
                </button>
              )}
            </div>
          </div>
        )}

        <div className="steps-overview">
          <h4>All Steps Overview</h4>
          <div className="steps-list">
            {currentRecipe.steps.map((step, index) => (
              <div
                key={step.step_number}
                className={`step-item ${completedSteps.has(step.step_number) ? 'completed' : ''} ${index === currentCookingStep ? 'current' : ''}`}
                onClick={() => setCurrentCookingStep(index)}
              >
                <div className="step-number">{step.step_number}</div>
                <div className="step-content">
                  <h5>{step.title}</h5>
                  <p>{step.description}</p>
                  {step.duration_minutes && <span className="duration">⏱️ {step.duration_minutes} min</span>}
                </div>
                {completedSteps.has(step.step_number) && <div className="checkmark">✅</div>}
              </div>
            ))}
          </div>
        </div>

        {allStepsCompleted && (
          <div className="completion-celebration">
            <div className="success-animation">
              <div className="pizza-completion">🍕✨</div>
              <h3>Congratulations! Your Pizza is Ready!</h3>
              <p>You've successfully completed all cooking steps. Enjoy your delicious custom pizza!</p>
              <button
                onClick={() => {
                  // Reset for new recipe
                  setCurrentStep(STEPS.INGREDIENTS);
                  setSelectedIngredients([]);
                  setDietaryPreferences({
                    diet_types: [],
                    additional_preferences: [],
                    spice_level: 0,
                    allergen_avoidance: []
                  });
                  setCurrentRecipe(null);
                  setCurrentCookingStep(0);
                  setCompletedSteps(new Set());
                }}
                className="start-over-button"
              >
                Create Another Pizza
              </button>
            </div>
          </div>
        )}

        <div className="recipe-details">
          <div className="sauce-preparation">
            <h4>Sauce Preparation</h4>
            <ul>
              {currentRecipe.sauce_preparation.map((step, index) => (
                <li key={index}>{step}</li>
              ))}
            </ul>
          </div>

          <div className="cooking-tips">
            <h4>Pro Tips</h4>
            <ul>
              {currentRecipe.tips.map((tip, index) => (
                <li key={index}>{tip}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className={`app ${darkMode ? 'dark' : ''}`}>
      <header className="app-header">
        <div className="header-content">
          <h1>🍕 AI Pizza Generator</h1>
          <div className="header-controls">
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="theme-toggle"
            >
              {darkMode ? '☀️' : '🌙'}
            </button>
          </div>
        </div>
        {renderStepProgress()}
      </header>

      <main className="main-content">
        {currentStep === STEPS.INGREDIENTS && renderIngredientSelection()}
        {currentStep === STEPS.PREFERENCES && renderDietaryPreferences()}
        {currentStep === STEPS.RELATED && renderRelatedRecipes()}
        {currentStep === STEPS.GENERATE && renderRecipeGeneration()}
        {currentStep === STEPS.COOKING && renderCookingAssistant()}
      </main>

      {renderConflictModal()}
    </div>
  );
}

export default App;