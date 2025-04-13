# self_healing_code/app/llm_module.py

import os
import openai
import logging
import json
import re # Ensure re is imported
from dotenv import load_dotenv
from typing import List # Import List type hint

# (Load env, logging setup, API key check remains the same)
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    logger.error("OPENAI_API_KEY not set.")
    raise EnvironmentError("Missing OPENAI_API_KEY")

# (get_variable_mapping remains the same - already requests functions/classes)
def get_variable_mapping(code: str, lang: str) -> dict:
    """Gets initial mapping for variables, functions, classes."""
    prompt = (
         f"You are an expert {lang} developer analyzing code for readability. "
         f"Given the following {lang} code, identify all variable names, function names, and class names "
         "that appear obfuscated (e.g., single-letter names like 'x', abbreviations like 'proc_dat', "
         "unclear names like 'tmp', or names with excessive underscores like 'val_'). "
         "Provide a JSON object mapping these original obfuscated names to more descriptive and conventional alternatives "
         f"following standard {lang} naming conventions (e.g., snake_case for Python variables/functions, PascalCase for Python classes). "
         "Focus only on renaming identifiers; do not change code structure, logic, or comments.\n\n"
         "Guidelines:\n"
         "- Choose clear, concise, and contextually appropriate names.\n"
         "- Be consistent with naming conventions for the language.\n"
         "- Map *only* the names you identify as needing improvement.\n"
         "- Return *only* a single, valid JSON object mapping original names (string keys) to suggested new names (string values).\n\n"
         f"Code:\n```\n{code}\n```\n\nJSON Mapping:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", # Or your preferred model
            messages=[
                {"role": "system", "content": f"You are a {lang} code analysis assistant generating improved identifier names in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, max_tokens=512, n=1,
            response_format={"type": "json_object"}
        )
        mapping_str = response.choices[0].message.content.strip()
        mapping_str = re.sub(r'^```json\s*|\s*```$', '', mapping_str).strip()
        if not mapping_str: raise ValueError("Empty response from LLM for initial mapping")
        logger.info(f"Raw mapping response for {lang}: {mapping_str}")
        mapping = json.loads(mapping_str)
        if not isinstance(mapping, dict): raise ValueError(f"LLM response is not a JSON object: {mapping_str}")
        cleaned_mapping = {str(k): str(v) for k, v in mapping.items() if isinstance(k, str) and isinstance(v, str) and k and v}
        logger.info(f"Obtained and cleaned identifier mapping for {lang}: {cleaned_mapping}")
        return cleaned_mapping
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error processing LLM mapping response: {e}\nRaw response: {mapping_str}")
        raise ValueError("Failed to decode LLM response into valid JSON.") from e
    except Exception as e:
        logger.error(f"Error obtaining identifier mapping: {e}")
        raise e

# (apply_mapping_to_code remains the same)
def apply_mapping_to_code(base_code: str, mapping: dict, lang: str) -> str:
    """Uses the OpenAI API to apply the given identifier mapping to the base_code."""
    mapping_json = json.dumps(mapping, indent=2)
    prompt = (
        f"You are an expert {lang} developer. Apply the following identifier mapping exactly to the code below. "
        "Do not alter any other identifiers, comments, or structure. Only change identifiers specified in the mapping. "
        "Return only the transformed code, ensuring it remains syntactically correct.\n\n"
        "Identifier Mapping:\n"
        f"{mapping_json}\n\n"
        "Original Code:\n"
        f"```\n{base_code}\n```\n\n"
        "Transformed Code Only:" # Be explicit
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", # Or preferred model
            messages=[
                {"role": "system", "content": f"You are a precise {lang} code transformation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Very low temp for precise application
            max_tokens=1024, # Allow sufficient length
            n=1
        )
        transformed_code = response.choices[0].message.content.strip()
        # Clean potential markdown fences if they appear
        transformed_code = re.sub(r'^```[a-zA-Z]*\s*', '', transformed_code)
        transformed_code = re.sub(r'\s*```$', '', transformed_code).strip()
        logger.info("Applied mapping to code successfully for %s.", lang)
        # logger.debug(transformed_code) # Log full code only in debug
        return transformed_code
    except Exception as e:
        logger.error("Error applying mapping to code: {e}")
        raise e


def suggest_better_names(original_name: str, current_name: str, code_context: str, lang: str, count: int = 3) -> List[str]:
    """
    Uses the OpenAI API to suggest MULTIPLE improved names for a given identifier,
    considering its context. Asks for diverse and potentially more descriptive options.
    Returns a list of strings.
    """
    prompt = (
        f"You are an expert {lang} developer focused on improving code readability and maintainability. "
        f"Analyze the identifier `{current_name}` (originally `{original_name}`) within the following {lang} code snippet:\n\n"
        f"Code Context:\n"
        f"```\n{code_context}\n```\n\n"
        f"Suggest exactly **{count}** potential alternative names for `{current_name}`. "
        "Prioritize names that are:\n"
        f"1. **More Descriptive:** Clearly convey the purpose based on the context.\n"
        f"2. **Conventional:** Follow standard {lang} naming conventions (e.g., snake_case/PascalCase for Python, camelCase/PascalCase for Java/C#, etc.).\n"
        f"3. **Concise but Clear:** Avoid unnecessary length but don't sacrifice clarity.\n"
        f"4. **Diverse:** Offer different stylistic or semantic alternatives if possible.\n\n"
        f"Even if `{current_name}` seems acceptable, try to offer improvements or valid alternatives.\n\n"
        "Return the suggestions as a JSON list of strings. Example: `[\"suggestion_one\", \"suggestion_two\", \"suggestion_three\"]`\n\n"
        "JSON List of Suggestions:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", # Or a more powerful model if needed
            messages=[
                {"role": "system", "content": f"You are a helpful assistant providing multiple improved {lang} identifier name suggestions as a JSON list."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5, # Higher temp for more diverse suggestions
            max_tokens=150, # Enough for a few names
            n=1,
            response_format={"type": "json_object"} # Request JSON
        )
        response_content = response.choices[0].message.content.strip()
        logger.debug(f"Raw name suggestions response for '{current_name}': {response_content}")

        # Expecting response like {"suggestions": ["name1", "name2"]} or just ["name1", "name2"]
        suggestions = []
        try:
            data = json.loads(response_content)
            if isinstance(data, list):
                 suggestions = data
            elif isinstance(data, dict):
                 # Look for a key that might contain the list (e.g., 'suggestions', 'names', 'alternatives')
                 possible_keys = ['suggestions', 'names', 'alternatives', 'results']
                 for key in possible_keys:
                     if key in data and isinstance(data[key], list):
                         suggestions = data[key]
                         break
                 if not suggestions: # If dict format but no expected key found
                      logger.warning(f"LLM returned JSON dict but no suggestion list found: {response_content}")

            # Validate suggestions are strings
            suggestions = [str(s) for s in suggestions if isinstance(s, str) and s]

        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON suggestions: {response_content}")
            # Attempt to extract names if it's just a simple list-like string without JSON structure
            # (e.g. "suggestion_one, suggestion_two") - less reliable
            suggestions = [name.strip() for name in response_content.split(',') if name.strip()]


        if not suggestions:
             logger.warning(f"LLM did not provide valid suggestions for '{current_name}'.")
             return [] # Return empty list

        logger.info(f"LLM suggested names for '{current_name}': {suggestions}")
        return suggestions[:count] # Return up to the requested count

    except Exception as e:
        logger.error(f"Error suggesting better names for '{current_name}': {e}")
        return [] # Return empty list on error


# (rename_variables_multilang can remain if needed for other modes)
def rename_variables_multilang(code: str, lang: str) -> str:
    """Uses the OpenAI API to directly transform code by renaming obfuscated variables."""
    # ... (existing implementation) ...
    prompt = (
        f"You are an expert {lang} developer. Transform the following {lang} code by renaming obfuscated variable names "
        "to more descriptive names, while preserving the code's structure and syntax exactly. "
        "Return only the transformed code without any additional commentary.\n\n"
        f"Original Code:\n```\n{code}\n```\n\nTransformed Code:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", messages=[{"role": "system", "content": f"You are a {lang} code transformation assistant."}, {"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=1024, n=1, stop=None
        )
        transformed_code = response.choices[0].message.content.strip()
        transformed_code = re.sub(r'^```[a-zA-Z]*\s*|\s*```$', '', transformed_code).strip()
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
        mapping = get_identifier_mapping(sample_code, lang)
        logger.info("Mapping: %s", mapping)
    except Exception as e:
        logger.error("Mapping failed: %s", e)
    
    try:
        transformed = rename_variables_multilang(sample_code, lang)
        logger.info("Transformed code: %s", transformed)
    except Exception as e:
        logger.error("Transformation failed: %s", e)

if __name__ == "__main__":
    # For testing purposes with a sample code snippet.
    sample_code = "def x(a, b): return a+b"
    lang = "python"
    try:
        mapping = get_identifier_mapping(sample_code, lang)
        logger.info("Mapping: %s", mapping)
    except Exception as e:
        logger.error("Mapping failed: %s", e)
    
    # Example usage of apply_mapping_to_code
    sample_mapping = {"x": "addFunction", "a": "valA", "b": "valB"}
    try:
        transformed = apply_mapping_to_code(sample_code, sample_mapping, lang)
        transformed = transformed.replace("```c", "").replace("```", "").strip()
        logger.info("Transformed code: %s", transformed)
    except Exception as e:
        logger.error("Transformation failed: %s", e)