# 🍕 AI Pizza Generator

An intelligent pizza recipe generator that uses AI to create custom pizza recipes based on your selected ingredients and dietary preferences. Built with React, FastAPI, and powered by OpenRouter's AI models.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![React](https://img.shields.io/badge/React-19.0.0-61DAFB?logo=react)
![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi)

## ✨ Features

- **Smart Ingredient Selection**: Choose from 7 categories including flours, cheeses, meats, vegetables, sauces, spices & herbs, and other toppings
- **Dietary Preference Support**: Vegan, vegetarian, gluten-free, keto, paleo, and 9+ other diet types
- **Allergen Detection**: Automatically detects conflicts between selected ingredients and dietary restrictions
- **AI-Powered Recipes**: Uses advanced AI models to generate unique, detailed pizza recipes
- **Step-by-Step Cooking Assistant**: Interactive cooking mode with built-in timers and progress tracking
- **Recipe Customization**: Adjust spice levels and set additional preferences (organic, non-GMO, etc.)
- **Related Recipe Suggestions**: Finds similar recipes based on your ingredient selections
- **Responsive Design**: Beautiful, modern UI with dark mode support

## 🏗️ Architecture

### Frontend
- **Framework**: React 19.0.0
- **Styling**: TailwindCSS 3.4.17
- **Build Tool**: CRACO (Create React App Configuration Override)
- **State Management**: React Hooks
- **HTTP Client**: Axios

### Backend
- **Framework**: FastAPI 0.104.1
- **Server**: Uvicorn 0.24.0
- **Database**: MongoDB Atlas (via Motor async driver)
- **AI Integration**: OpenRouter API (DeepSeek model)
- **Authentication**: Environment-based API key management

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.11.9
- MongoDB Atlas account
- OpenRouter API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AI-PIZZA-GENERATOR.git
   cd AI-PIZZA-GENERATOR
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Create environment file**
   Create a `.env` file in the `backend` directory:
   ```env
   LLAMA_API_KEY=your_openrouter_api_key
   LLAMA_API_BASE_URL=https://openrouter.ai/api/v1
   LLAMA_MODEL=deepseek/deepseek-r1-0528-qwen3-8b:free
   LLAMA_TIMEOUT=30
   MONGODB_URI=mongodb+srv://your_username:your_password@your_cluster.mongodb.net/pizza_generator
   ```

4. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   # or
   yarn install
   ```

5. **Run the application**

   **Backend** (from project root):
   ```bash
   cd backend
   uvicorn server:app --reload --host 0.0.0.0 --port 8000
   ```

   **Frontend** (from project root):
   ```bash
   cd frontend
   npm start
   # or
   yarn start
   ```

   The frontend will be available at `http://localhost:3000` and the backend API at `http://localhost:8000`

## 📁 Project Structure

```
AI-PIZZA-GENERATOR/
├── backend/
│   ├── server.py              # FastAPI application
│   ├── requirements.txt      # Python dependencies
│   ├── runtime.txt           # Python version specification
│   └── backend_test.py       # Backend tests
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── App.css          # Component styles
│   │   ├── index.js         # React entry point
│   │   └── index.css        # Global styles
│   ├── public/               # Static assets
│   ├── build/               # Production build
│   ├── package.json         # Node dependencies
│   ├── tailwind.config.js   # TailwindCSS configuration
│   └── craco.config.js      # CRACO configuration
├── tests/                   # Test files
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔧 API Endpoints

### Health Check
- `GET /` - API status
- `GET /health` - Health check with database connection

### Ingredients
- `GET /api/ingredients` - Get all ingredient categories
- `GET /api/ingredients/{category}` - Get ingredients by category

### Recipe Generation
- `POST /api/generate-recipe` - Generate a custom recipe
- `POST /api/check-conflicts` - Check for dietary conflicts
- `POST /api/find-related-recipes` - Find similar recipes

### Progress Tracking
- `POST /api/save-cooking-progress` - Save cooking step progress

## 🧪 Testing

### Backend Tests
```bash
cd backend
python backend_test.py
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 🌐 Deployment

### Frontend (Vercel)
1. Connect your GitHub repository to Vercel
2. Set build command: `cd frontend && npm run build`
3. Set output directory: `frontend/build`
4. Add environment variable: `REACT_APP_API_BASE_URL`

### Backend (Render)
1. Connect your GitHub repository to Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
4. Add environment variables from `.env` file

## 🔒 Security

- API keys are stored in environment variables
- MongoDB connection uses TLS encryption
- CORS is configured for specific origins
- No sensitive data is committed to the repository
- `.gitignore` excludes all environment files

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- OpenRouter for providing AI API access
- MongoDB Atlas for database hosting
- The React and FastAPI communities

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for pizza lovers everywhere**