# self_healing_code/app/llm_module.py

import os
import openai
import logging
import json
import re
from dotenv import load_dotenv
from typing import List # Import List type hint

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    logger.error("OPENAI_API_KEY not set.")
    raise EnvironmentError("Missing OPENAI_API_KEY")

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
    
def deobfuscate_strings_llm(code: str, lang: str, **kwargs) -> str | None:
    """
    Uses an LLM to identify and deobfuscate common string obfuscations in code.

    Args:
        code: The source code string.
        lang: The programming language identifier.
        **kwargs: Additional arguments for the LLM call (e.g., model, temperature).

    Returns:
        The code string potentially modified with deobfuscated strings,
        or None if an error occurs during the LLM call.
        Returns the original code string if the LLM indicates no changes were made.
    """
    logger.info(f"Requesting LLM for string deobfuscation (lang: {lang})...")

    common_patterns_desc = """
    - Character Arrays/Lists initialized with integer char codes (e.g., `char s[] = {104, 101, ...};`, `let s = [104, 101, ...]`, `s = [104, 101, ...]`).
    - Hex Arrays/Strings (e.g., `0x68, 0x65, ...`).
    - Base64 Encoded Strings, often used with decode functions (e.g., `base64.b64decode('aGVsbG8=')`).
    - Concatenation of multiple small strings or single characters.
    - Simple XOR/Shift encoded strings (look for decoding loops/functions).
    - Calls to custom decoding functions (e.g., `decode_string`, `get_hidden_text`).
    - Excessive octal or hex escapes within string literals.
    """

    prompt = f"""
        You are an expert code analysis assistant specializing in deobfuscation for the '{lang}' language.
        Your task is to identify and reverse common string obfuscation techniques within the provided code snippet.

        Analyze the following '{lang}' code:
        {code}

        Identify instances where string literals appear to be obfuscated using techniques such as (but not limited to):
        {common_patterns_desc}

        Replace these obfuscated representations *in place* with their original, plain string literal equivalents, formatted correctly for '{lang}'.

        **Instructions:**
        1.  Carefully analyze and decode/reassemble obfuscated strings.
        2.  Modify the code to replace the obfuscated form with the plain string literal (e.g., replace `char s[] = {{104, 101, ...}};` with `char* s = "hello";` or `print(b64decode('aGVsbG8='))` with `print("hello")`, adjusting for '{lang}' syntax).
        3.  **Only modify the string representations.** Do NOT change variable names, logic, control flow, comments, etc., unless removing a decoding mechanism made redundant by the inline string.
        4.  If an obfuscation method is complex or unclear, **leave it unchanged**. Prioritize correctness.
        5.  Return the **entire modified code block** as a single response, without explanations or markdown formatting.

        Modified Code Only:
        """

    try:
        llm_model = kwargs.get("llm_model", "gpt-4o-mini")
        temperature = kwargs.get("temperature", 0.2)

        response = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": f"You are a code deobfuscation assistant for {lang} focused on strings. You only output modified code."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max(2048, len(code.split()) + 512),
            n=1,
            stop=None
        )

        transformed_code = response.choices[0].message.content.strip()
        # Clean LLM Output
        transformed_code = re.sub(r'^```[a-zA-Z]*\s*', '', transformed_code)
        transformed_code = re.sub(r'\s*```$', '', transformed_code).strip()

        if not transformed_code:
            logger.warning("LLM returned empty response for string deobfuscation.")
            # Return original code if LLM gives up or fails in a way that results in empty output
            return code
        else:
            # Return potentially modified code (could be same as original if no changes made)
            return transformed_code

    except Exception as e:
        logger.error(f"LLM call failed during string deobfuscation: {e}", exc_info=True)
        return None # Indicate critical failure in LLM communication/processing

def simplify_control_flow_llm(code: str, lang: str, active_simplification_flags: List[str], **kwargs) -> str | None:
    """
    Uses an LLM to apply specific, requested control flow simplifications to code.

    Args:
        code: The source code string.
        lang: The programming language identifier.
        active_simplification_flags: A list of strings specifying which simplifications
                                      the LLM should attempt (e.g., from SIMPLIFICATION_FLAGS).
        **kwargs: Additional arguments for the LLM call (e.g., model, temperature).

    Returns:
        The code string potentially modified with simplified control flow,
        or None if a critical error occurs during the LLM call.
        Returns the original code string if the LLM indicates no changes were made or possible.
    """
    if not active_simplification_flags:
        logger.debug("No active control flow simplification flags provided. Skipping LLM call.")
        return code # Return original code if no flags are active

    logger.info(f"Requesting LLM for control flow simplification (lang: {lang}, flags: {active_simplification_flags})...")

    # Construct the prompt dynamically based on active flags
    flags_string = chr(10).join(f"- {flag}" for flag in active_simplification_flags) # Use newline join

    prompt = f"""
        You are an expert code refactoring assistant for the '{lang}' language, focused ONLY on specific control flow simplifications.
        Analyze the following '{lang}' code:
        {code}


        Apply **ONLY** the following types of control flow simplifications where applicable and **semantically equivalent**:
        {flags_string}

        **Important Instructions:**
        1.  Apply *only* the requested simplifications listed above. Do NOT apply other simplifications.
        2.  **Crucially: Ensure the logical behavior of the code remains IDENTICAL.** Do not make changes if unsure about equivalence. This is the highest priority.
        3.  Do NOT perform other refactorings (like renaming, constant folding, string changes, etc.).
        4.  If none of the requested simplifications can be safely applied, return the original code block unchanged.
        5.  Return the **entire modified code block** as a single response. Do not add explanations or markdown formatting. Just the raw code.

        Modified Code Only:
        """

    try:
        llm_model = kwargs.get("llm_model", "gpt-4o-mini") # Use a capable model
        temperature = kwargs.get("temperature", 0.1) # Very low temp for precise, safe changes

        response = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": f"You are a precise code refactoring assistant for {lang} focused only on specified control flow simplifications. You only output code."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max(2048, len(code.split()) + 512), # Ensure enough tokens
            n=1,
            stop=None
        )

        simplified_code = response.choices[0].message.content.strip()
        # Clean LLM Output
        simplified_code = re.sub(r'^```[a-zA-Z]*\s*', '', simplified_code)
        simplified_code = re.sub(r'\s*```$', '', simplified_code).strip()

        if not simplified_code:
            logger.warning("LLM returned empty response for control flow simplification.")
            return code # Return original code if LLM gives up
        else:
             # Return potentially modified code (could be same as original if no changes applied)
             # Let the calling function compare if needed
             return simplified_code

    except Exception as e:
        logger.error(f"LLM call failed during control flow simplification: {e}", exc_info=True)
        return None 
    
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
    
def simplify_expressions_llm(code: str, lang: str, **kwargs) -> str | None:
    """
    Uses an LLM to simplify constant expressions and boolean logic in code.

    Args:
        code: The source code string.
        lang: The programming language identifier.
        **kwargs: Additional arguments for the LLM call (e.g., model, temperature).

    Returns:
        The code string potentially modified with simplified expressions,
        or None if a critical error occurs during the LLM call.
        Returns the original code string if the LLM indicates no changes were made.
    """
    logger.info(f"Requesting LLM for expression simplification (lang: {lang})...")

    simplification_types = """
    - Constant Folding: Evaluate constant arithmetic expressions (e.g., `2 + 3 * 4` -> `14`).
    - Arithmetic Identities: Simplify operations like `x + 0`, `x - 0`, `x * 1`, `x / 1`, `x * 0`, `0 / x`.
    - Boolean Logic: Simplify expressions using standard rules (De Morgan's laws, double negation, identity laws, etc.). E.g., `!(!a || !b)` -> `a && b`.
    - Trivial Conditions: Simplify comparisons involving constants like `x == true` (if language allows) -> `x`, `y == false` -> `!y`, etc.
    - Redundant Operations: Remove redundant casts or operations where possible.
    """

    prompt = f"""
        You are an expert code optimization assistant for the '{lang}' language, focusing ONLY on simplifying expressions (constants and boolean logic).
        Analyze the following '{lang}' code:
        {code}


        Apply the following types of expression simplifications where applicable and **semantically equivalent**:
        {simplification_types}

        **Instructions:**
        1.  Apply *only* the requested expression simplifications.
        2.  **Crucially: Ensure the logical behavior of the code remains IDENTICAL.** Do not simplify if it changes program logic (e.g., be careful with floating-point precision issues or potential side effects in expressions if applicable to '{lang}').
        3.  Do NOT perform other refactorings (like renaming, string changes, control flow changes, dead code removal).
        4.  If no simplifications can be safely applied, return the original code block unchanged.
        5.  Return the **entire modified code block** as a single response. Do not add explanations or markdown formatting. Just the raw code.

        Modified Code Only:
        """

    try:
        llm_model = kwargs.get("llm_model", "gpt-4o-mini")
        temperature = kwargs.get("temperature", 0.1) # Low temp for precise changes

        response = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": f"You are a precise code optimization assistant for {lang} focused only on expression simplification. You only output code."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max(2048, len(code.split()) + 512),
            n=1,
            stop=None
        )

        simplified_code = response.choices[0].message.content.strip()
        # Clean LLM Output
        simplified_code = re.sub(r'^```[a-zA-Z]*\s*|\s*```$', '', simplified_code).strip()

        if not simplified_code:
            logger.warning("LLM returned empty response for expression simplification.")
            return code # Return original
        else:
            return simplified_code # Return potentially modified code

    except Exception as e:
        logger.error(f"LLM call failed during expression simplification: {e}", exc_info=True)
        return None # Indicate critical failure
    
def remove_dead_code_llm(code: str, lang: str, **kwargs) -> str | None:
    """
    Uses an LLM to cautiously identify and remove dead/unreachable code.

    Args:
        code: The source code string.
        lang: The programming language identifier.
        **kwargs: Additional arguments for the LLM call (e.g., model, temperature).

    Returns:
        The code string potentially modified with dead code removed,
        or None if a critical error occurs during the LLM call.
        Returns the original code string if the LLM indicates no changes were made or possible.
    """
    logger.info(f"Requesting LLM for dead code removal (lang: {lang})...")

    # Emphasize safety and specific types of dead code
    dead_code_types = """
    - Unused Local Variables: Variables declared within a function/scope but never read or used.
    - Unused Private/Static Functions: Functions not exported and never called from within the analyzed code block (be cautious of external calls, reflection, etc.).
    - Unreachable Code: Code immediately following unconditional return, break, continue, or throw statements within the same block.
    - Redundant/No-Operation Code: Statements that have no effect.
    """

    prompt = f"""
        You are an expert code analysis assistant for the '{lang}' language, focused **ONLY** on **safely removing obviously dead or unreachable code**.
        Analyze the following '{lang}' code:
        {code}


        Identify and remove code constructs that are **provably** dead or unreachable based *only* on the provided code context. Focus on:
        {dead_code_types}

        **EXTREMELY IMPORTANT SAFETY INSTRUCTIONS:**
        1.  **BE CONSERVATIVE:** If there is **ANY** doubt about whether code is truly dead (e.g., a function might be called externally, a variable used via reflection or dynamic means), **DO NOT REMOVE IT**.
        2.  **PRESERVE BEHAVIOR:** The primary goal is to remove useless code **WITHOUT** changing the program's observable behavior or logic in any way.
        3.  **NO OTHER CHANGES:** Do NOT perform any other refactoring (renaming, simplification, formatting, etc.). Only remove code identified as dead/unreachable according to the strict rules above.
        4.  **FUNCTIONS/METHODS:** Be extra cautious removing functions or methods. Only remove clearly unused **private** or **static** functions/methods that are not part of any public API or interface definition. Leave public/exported functions untouched unless you are absolutely certain they are unused *and* not part of an external contract.
        5.  If no code can be safely removed, return the original code block unchanged.
        6.  Return the **entire modified code block** as a single response. Do not add explanations or markdown formatting. Just the raw code.

        Modified Code Only:
        """

    try:
        # Use a model known for strong reasoning, maybe slightly higher temp might explore *what* could be dead, but safety demands caution. Let's keep temp low.
        llm_model = kwargs.get("llm_model", "gpt-4o-mini") # Consider GPT-4 for better reasoning if needed
        temperature = kwargs.get("temperature", 0.1) # Very low temp for safety

        response = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": f"You are a highly cautious code analysis assistant for {lang} focused *only* on safely removing dead code. You only output code."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max(2048, len(code.split()) + 256), # Allow some shrinkage, but still need buffer
            n=1,
            stop=None
        )

        cleaned_code = response.choices[0].message.content.strip()
        # Clean LLM Output
        cleaned_code = re.sub(r'^```[a-zA-Z]*\s*|\s*```$', '', cleaned_code).strip()

        if not cleaned_code:
            logger.warning("LLM returned empty response for dead code removal.")
            return code # Return original code if LLM gives up
        else:
            return cleaned_code # Return potentially modified code

    except Exception as e:
        logger.error(f"LLM call failed during dead code removal: {e}", exc_info=True)
        return None # Indicate critical failure
    