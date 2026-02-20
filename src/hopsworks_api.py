"""
Singleton Hopsworks connection manager.
Prevents repeated logins and manages the project instance.
"""
import os
import sys
import hopsworks
from dotenv import load_dotenv

# Ensure environment is loaded from project root
# Assuming this file is in src/hopsworks_api.py -> need to go up to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(ENV_PATH)

_project = None

def get_project():
    """Returns a singleton Hopsworks project instance."""
    global _project
    
    if _project is not None:
        return _project

    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME")

    if not api_key:
        print("❌ Error: HOPSWORKS_API_KEY not found in environment.")
        return None

    try:
        print(f"🔌 Connecting to Hopsworks project '{project_name}'...")
        # Login (hopsworks library handles token caching usually, but we keep instance in memory)
        _project = hopsworks.login(project=project_name, api_key_value=api_key)
        print("✅ Connected to Hopsworks.")
    except Exception as e:
        print(f"❌ Failed to connect to Hopsworks: {e}")
        # Return None or re-raise depending on strictness. Returning None allows caller to handle.
        return None

    return _project
