"""
Gemini API Test Script
Tests API key configuration and lists available models using the new google-genai SDK.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from google import genai


def test_api():
    """Test Gemini API connection and list available models."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    
    if not api_key or api_key == "your_api_key_here":
        print("GEMINI_API_KEY not found or not set in .env file")
        print("Get a free key at: https://aistudio.google.com/apikey")
        return False
    
    print("Testing Google Gemini API (google-genai SDK)...")
    print()
    
    try:
        client = genai.Client(api_key=api_key)
        print("✅ API key configured successfully")
        print()
        
        print("Listing available text generation models:")
        print("-" * 50)
        
        count = 0
        for model in client.models.list():
            methods = getattr(model, 'supported_generation_methods', None) or []
            methods = [str(m) for m in methods]
            if 'generateContent' in methods:
                name = getattr(model, 'name', 'unknown')
                display = getattr(model, 'display_name', name)
                print(f"✓ {name}")
                print(f"  Display Name: {display}")
                print()
                count += 1
        
        print(f"Found {count} models supporting generateContent")
        print()
        
        # Quick translation test
        print("Running quick translation test...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents='Translate this Japanese to English (just the translation, nothing else): "こんにちは"',
        )
        print(f"[OK] Translation test: konnichiwa -> {response.text.strip()}")
        print()
        print("API is working correctly!")
        
        return True
        
    except Exception as e:
        print(f" Error: {e}")
        print()
        print("This might mean:")
        print("1. API key is invalid or expired")
        print("2. API key doesn't have proper permissions")
        print("3. Network/firewall blocking the API")
        print("4. Regional restrictions")
        return False


if __name__ == "__main__":
    test_api()
