# CrewAI Laptop Recommendation System

A web-based application that uses CrewAI and FastAPI to recommend laptops based on a student or professional profile, budget, and hardware preferences. The system works with the local laptop dataset and can also use an LLM provider when available.

## Features
- AI-powered laptop recommendations using CrewAI
- Local fallback recommendations when the LLM provider is unavailable
- Dataset-based filtering and stats from the cleaned laptop CSV
- FastAPI backend with a simple frontend dashboard

## Run Locally
1. Activate the virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the backend:
   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```
4. Open http://127.0.0.1:8000

## Notes
- The app can work with or without an LLM provider.
- If the provider fails, it falls back to dataset-based recommendations.
- For real AI-generated summaries, configure an API key for OpenAI or Gemini.
