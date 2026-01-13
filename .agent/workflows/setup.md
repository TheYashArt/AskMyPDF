---
description: Initial project setup and environment configuration
---

# Setup Workflow

This workflow guides you through setting up the environment and configurations required to run the `AskMyPdf` project.

## 1. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

// turbo
```powershell
python -m venv venv
.\venv\Scripts\activate
```

## 2. Install Dependencies
Install all required Python packages from `requirements.txt`.

// turbo
```powershell
pip install -r requirements.txt
```

## 3. Configure API Keys
The project requires API keys for Groq and Gemini.

1. Create a `.streamlit/secrets.toml` file in the project root if it doesn't exist.
2. Add your API keys to the file:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GEMINI_API_KEY = "your_gemini_api_key_here"
```

> [!IMPORTANT]
> Never commit your `secrets.toml` file to version control.
