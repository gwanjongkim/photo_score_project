import os

print("Listing available Gemini models that support vision...")

try:
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        exit(1)

    genai.configure(api_key=api_key)

    found_models = []
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            # Simple heuristic to find vision-capable models
            if 'vision' in model.name or 'flash' in model.name:
                found_models.append(model.name)

    if found_models:
        print("Found the following suitable models:")
        for model_name in sorted(found_models):
            print(f"- {model_name}")
    else:
        print("No suitable vision models found in the current environment.")

except Exception as e:
    print(f"An error occurred: {e}")
