# self_healing_code/app/llm_module.py

import os
import openai
import logging
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    logger.error("OPENAI_API_KEY not set. Please check your .env file.")
    raise EnvironmentError("Missing OPENAI_API_KEY")

def get_variable_mapping(code: str, lang: str) -> dict:
    """
    Uses the OpenAI API to obtain a mapping from obfuscated variable names to descriptive names
    for code in the specified language.
    """
    prompt = (
        f"You are an expert {lang} developer. Given the following {lang} code, "
        "identify all variable names that appear obfuscated (e.g., single-letter or unclear names), "
        "and provide a JSON object mapping these variable names to more descriptive alternatives. "
        "Do not change any code structure or punctuation. Return only a valid JSON object.\n\n"
        "Code:\n"
        f"{code}\n\n"
        "Mapping:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="o3-mini-high",
            messages=[
                {"role": "system", "content": f"You are a {lang} code analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=256,
            n=1,
            stop=None
        )
        mapping_str = response.choices[0].message.content.strip()
        mapping = json.loads(mapping_str)
        logger.info("Obtained variable mapping for %s: %s", lang, mapping)
        return mapping
    except Exception as e:
        logger.error("Error obtaining variable mapping: %s", e)
        raise e

def rename_variables_multilang(code: str, lang: str) -> str:
    """
    Uses the OpenAI API to directly transform the given code (in any supported language)
    by renaming obfuscated variables to more descriptive names, ensuring the code's structure and syntax remain unchanged.
    """
    prompt = (
        f"You are an expert {lang} developer. Transform the following {lang} code by renaming obfuscated variable names "
        "to more descriptive names, while preserving the code's structure and syntax exactly. "
        "Return only the transformed code without any additional commentary.\n\n"
        "Original Code:\n"
        f"{code}\n\n"
        "Transformed Code:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are a {lang} code transformation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024,
            n=1,
            stop=None
        )
        transformed_code = response.choices[0].message.content.strip()
        logger.info("Code transformation successful for %s.", lang)
        return transformed_code
    except Exception as e:
        logger.error("Error during code transformation: %s", e)
        raise e

if __name__ == "__main__":
    # For testing purposes with a sample code snippet.
    sample_code = "def x(a, b): return a+b"
    lang = "python"
    try:
        mapping = get_variable_mapping(sample_code, lang)
        logger.info("Mapping: %s", mapping)
    except Exception as e:
        logger.error("Mapping failed: %s", e)
    
    try:
        transformed = rename_variables_multilang(sample_code, lang)
        logger.info("Transformed code: %s", transformed)
    except Exception as e:
        logger.error("Transformation failed: %s", e)
