import os
import openai
import logging
import json
import re
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
    Uses the OpenAI API to obtain a mapping from obfuscated identifiers (variables, functions, potentially classes)
    to descriptive names for code in the specified language, returning a dict.
    """
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
        "Code:\n"
        f"```\n{code}\n```\n\n"
        "JSON Mapping:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", # Or your preferred model
            messages=[
                {"role": "system", "content": f"You are a {lang} code analysis assistant designed to suggest improved identifier names in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Low temp for predictability
            max_tokens=512, # Adjust as needed
            n=1,
            response_format={"type": "json_object"} # Request JSON output if using compatible models/API versions
        )
        mapping_str = response.choices[0].message.content.strip()

        # Clean potential markdown/formatting issues if response_format isn't perfect
        mapping_str = re.sub(r'^```json\s*', '', mapping_str)
        mapping_str = re.sub(r'\s*```$', '', mapping_str)
        mapping_str = mapping_str.strip()

        if not mapping_str:
            raise ValueError("Empty response from LLM for initial mapping")

        logger.info(f"Raw mapping response for {lang}: {mapping_str}")
        mapping = json.loads(mapping_str)

        # Basic validation/cleaning
        if not isinstance(mapping, dict):
            raise ValueError(f"LLM response is not a JSON object: {mapping_str}")
        # Ensure keys/values are strings and filter empty suggestions
        cleaned_mapping = {str(k): str(v) for k, v in mapping.items() if isinstance(k, str) and isinstance(v, str) and k and v}

        logger.info(f"Obtained and cleaned identifier mapping for {lang}: {cleaned_mapping}")
        return cleaned_mapping
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error processing LLM mapping response: {e}\nRaw response: {mapping_str}")
        raise ValueError("Failed to decode LLM response into valid JSON.") from e
    except Exception as e:
        logger.error(f"Error obtaining identifier mapping: {e}")
        raise e # Re-raise

def suggest_better_name(original_name: str, current_name: str, code_context: str, lang: str) -> str:
    """
    Uses the OpenAI API to suggest a single, improved name for a given identifier,
    considering its original name, current name, and code context.
    """
    prompt = (
        f"You are an expert {lang} developer focused on improving code readability. "
        f"I need a better name for an identifier in a {lang} code snippet.\n\n"
        f"- Original obfuscated name (if known): `{original_name}`\n"
        f"- Current name in the code: `{current_name}`\n"
        f"- Context (surrounding code snippet):\n"
        f"```\n{code_context}\n```\n\n"
        f"Analyze the context and the current name (`{current_name}`). Suggest a *single*, more descriptive, and conventional "
        f"identifier name following {lang} best practices (e.g., snake_case for Python vars/funcs, PascalCase for classes). "
        f"The new name should be a clear improvement over `{current_name}`.\n\n"
        f"Consider the identifier's likely role based on the code context. If `{current_name}` is already good, you can return it.\n\n"
        "Constraints:\n"
        "- Return ONLY the suggested identifier name as a single string.\n"
        "- Do not include backticks, quotes, or any explanation.\n"
        "- The name must be a valid identifier in {lang}.\n\n"
        "Suggested Name:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", # Use a capable model
            messages=[
                {"role": "system", "content": f"You are a helpful assistant providing improved {lang} identifier names."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4, # Allow a bit more creativity than initial mapping
            max_tokens=60,   # Name suggestion shouldn't be long
            n=1,
            stop=None # Let it generate the name
        )
        suggested_name = response.choices[0].message.content.strip()

        # Clean the suggestion: remove surrounding quotes or backticks if any
        suggested_name = re.sub(r'^[`"\']', '', suggested_name)
        suggested_name = re.sub(r'[`"\']$', '', suggested_name)
        suggested_name = suggested_name.strip() # Remove whitespace

        # Basic validation (can be enhanced)
        if not suggested_name or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_\-]*$', suggested_name):
             logger.warning(f"Received invalid name suggestion: '{suggested_name}' for current name '{current_name}'. Falling back to current name.")
             return current_name # Return current name if suggestion is invalid

        logger.info(f"LLM suggested name '{suggested_name}' for current name '{current_name}' (original: '{original_name}')")
        return suggested_name

    except Exception as e:
        logger.error(f"Error suggesting better name for '{current_name}': {e}")
        # Fallback to the current name in case of error
        return current_name

def apply_mapping_to_code(base_code: str, mapping: dict, lang: str) -> str:
    """
    Uses the OpenAI API to apply the given identifier mapping (variables and functions)
    to the base_code exactly as provided.

    :param base_code: Original obfuscated code.
    :param mapping: Dict of { original_ident: new_ident, ... } describing how to rename identifiers.
    :param lang: Language string (e.g. 'python', 'c', 'php', etc.).
    :return: Transformed code with the mapping applied.
    """
    mapping_json = json.dumps(mapping, indent=2)

    prompt = (
        f"You are an expert {lang} developer. Apply the following identifier mapping (for variables and functions) exactly to the code below. "
        "Rename only the specified identifiers. Do not alter any other identifiers, comments, structure, logic, or syntax. "
        "Return ONLY the fully transformed code, nothing else.\n\n"
        "Identifier Mapping:\n"
        f"{mapping_json}\n\n"
        "Original Code:\n"
        f"{base_code}\n\n"
        "Transformed Code:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", # Using a capable model for accurate transformation
            messages=[
                {"role": "system", "content": f"You are a precise {lang} code transformation assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, # Zero temperature for deterministic application
            max_tokens=1536, # Allow more tokens for larger code blocks
            n=1
        )
        transformed_code = response.choices[0].message.content.strip()

        # Basic cleanup to remove potential markdown fences
        transformed_code = re.sub(r'^```[a-zA-Z]*\n?', '', transformed_code)
        transformed_code = re.sub(r'\n?```$', '', transformed_code)

        logger.info("Applied mapping to code successfully for %s.", lang)
        logger.info("Transformed code:\n%s", transformed_code) # Use debug level for potentially large output
        return transformed_code
    except Exception as e:
        logger.error("Error applying mapping to code: %s", e)
        raise e

def rename_variables_multilang(code: str, lang: str) -> str:
    """
    Uses the OpenAI API to directly transform the given code (in any supported language)
    by renaming obfuscated identifiers (variables and functions) to more descriptive names,
    ensuring the code's structure and syntax remain unchanged.
    """
    prompt = (
        f"You are an expert {lang} developer. Transform the following {lang} code by renaming obfuscated identifier names "
        "(variables and functions like 'x', 'a', 'f1', 'tempVar') to more descriptive names based on their context and usage. "
        "Preserve the code's structure, logic, comments, and syntax exactly. "
        "Return ONLY the transformed code without any additional commentary or explanation.\n\n"
        "Original Code:\n"
        f"{code}\n\n"
        "Transformed Code:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini", # Using a capable model
            messages=[
                {"role": "system", "content": f"You are a {lang} code transformation assistant focused on renaming."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # Low temperature for consistency
            max_tokens=1536, # Allow more tokens
            n=1,
            stop=None
        )
        transformed_code = response.choices[0].message.content.strip()

        # Basic cleanup
        transformed_code = re.sub(r'^```[a-zA-Z]*\n?', '', transformed_code)
        transformed_code = re.sub(r'\n?```$', '', transformed_code)

        logger.info("Code transformation (direct renaming) successful for %s.", lang)
        return transformed_code
    except Exception as e:
        logger.error("Error during direct code transformation: %s", e)
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