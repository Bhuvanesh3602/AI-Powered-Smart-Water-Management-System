# AI-Powered Smart Water Management & Prediction System (AquaMind)
<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/ff8ae2df-bb51-4cf4-8455-8eea576e5f70" />


AquaMind is a compact research/demo project that collects, analyzes and predicts water usage patterns and provides actionable suggestions using a small machine-learning predictor and optional AI assistant (Gemini/Google Generative Language). The project includes simple data management, anomaly detection, forecasting/prediction logic, and Streamlit-based UI for visualization and interaction.

This repository contains both a top-level lightweight script set and a package-style folder `SmartWaterPredictor/` containing the same modules for use as a library.

## Key features

- Load and normalize water usage datasets (CSV).  
- Compute usage statistics, daily/ hourly aggregates, and basic anomaly detection.  
- A simple predictor for short-term water usage forecasting
- Streamlit UI (`app.py`) to explore data, labels, anomalies, and predictions.  
- Optional integration with the Google Generative Language (Gemini) API to provide AI-driven tips and explanations (via `SmartWaterPredictor/gemini_helper.py`).

## Repository layout

- `app.py` - Streamlit application (UI and interaction).  
- `main.py` - Script entry point (alternative runner).  
- `data_manager.py` / `SmartWaterPredictor/data_manager.py` - dataset loading and utilities.  
- `predictor.py` / `SmartWaterPredictor/predictor.py` - prediction logic and model wrapper.  
- `anomaly_detector.py` / `SmartWaterPredictor/anomaly_detector.py` - simple anomaly detection helpers.  
- `SmartWaterPredictor/gemini_helper.py` - small REST wrapper to call the Gemini generateContent endpoint
- `pyproject.toml` - project metadata.  
- `attached_assets/` - example assets and prompt files.

## Requirements

Recommended Python version: 3.9+ (works on 3.10/3.11). Install dependencies in a virtual environment.

A minimal list of Python packages used by the project (install with pip):

- streamlit
- pandas
- numpy
- requests
- kagglehub (optional, for dataset download)
- python-dotenv (optional, for loading .env files)

If you prefer, create a `requirements.txt` in the project root with these packages and run `pip install -r requirements.txt`.

## Setup (Windows / PowerShell)

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

2. Install dependencies (example):

```powershell
pip install streamlit pandas numpy requests python-dotenv
# optional: kagglehub if you plan to use dataset downloading feature
pip install kagglehub
```

3. (Optional) Create a `.env` file in the project root to store environment variables during development:

```
GEMINI_API_KEY=YOUR_ACTUAL_KEY_HERE
```

Then load it in your environment (if you use `python-dotenv`, call `load_dotenv()` early in your app), or use the PowerShell method below.

## Setting the Gemini / Google API key

The Gemini helper reads the API key from the environment. It will use either `GEMINI_API_KEY` or `GOOGLE_API_KEY` (the code checks `GEMINI_API_KEY` first and falls back to `GOOGLE_API_KEY`).

Temporary (current PowerShell session only):

```powershell
$env:GEMINI_API_KEY = "YOUR_API_KEY_HERE"
# verify
python -c "import os; print(os.getenv('GEMINI_API_KEY'))"
```

Persistent (user-level):

```powershell
setx GEMINI_API_KEY "YOUR_API_KEY_HERE"
# Close and re-open your terminal to pick up the new variable
```

VS Code launch config: add an `env` block to `.vscode/launch.json` for the run/debug configuration.

Security notes:
- Do NOT commit API keys to source control. Use environment variables or a secrets manager.  
- For initial debugging, you can create a test API key in Google Cloud Console (make sure the Generative Language API is enabled for your project). If you get an "API key not valid" error, check that the API is enabled and that any key restrictions (IP, referrer) aren't blocking your call.

## Running the app

From the project root (PowerShell):

Streamlit UI (recommended):

```powershell
# in a shell where GEMINI_API_KEY is set if you plan to use the AI features
streamlit run app.py
```

Run via Python (non-Streamlit runner):

```powershell
python main.py
```

## Example quick test for Gemini helper

After setting `GEMINI_API_KEY` in your environment, run the following to confirm the helper sees it:

```powershell
$env:GEMINI_API_KEY = "YOUR_API_KEY"
python - <<'PY'
from SmartWaterPredictor import gemini_helper
print('Gemini client key loaded:', gemini_helper.get_client().api_key is not None)
PY
```

If the call prints `True` the key is visible to Python; actual requests to the Gemini API will still require a valid key and a project with the Generative Language API enabled.

## Troubleshooting

- Error: "API key not valid. Please pass a valid API key."  
  - Confirm you set `GEMINI_API_KEY` or `GOOGLE_API_KEY`.  
  - Confirm the key was created in Google Cloud Console and the Generative Language API is enabled for the project.  
  - Temporarily remove restrictions on the key (IP/referrer) to test.  

- Streamlit errors about invalid arguments (e.g., width param):  
  - Some older code may pass invalid types (e.g., `width='stretch'`) to Streamlit; prefer `use_container_width=True` when calling `st.dataframe` or pass an integer width in pixels.

- Missing packages: run `pip install` for the packages listed above.

## Development notes & contributions

- The project follows a simple script/module pattern for experimentation.  
- If you make changes that require new dependencies, add them to `pyproject.toml` or provide a `requirements.txt`.  
- Keep secrets and keys out of the repository; use environment variables or a secrets manager for production.

If you'd like, I can:
- Add a `requirements.txt` automatically from the currently used packages.  
- Add a `.env.sample` example and code to auto-load it with `python-dotenv`.  
- Wire up a small unit test or CI check for imports.

## License

This repository does not include a license file. If you plan to publish the project, add a LICENSE file (MIT, Apache 2.0, etc.).

---

Project created for experimentation and demonstration of combining simple analytics with optional generative AI assistance for water usage insights.
