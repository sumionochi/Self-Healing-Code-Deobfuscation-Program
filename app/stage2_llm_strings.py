# stage2_llm_strings.py

import logging
import re
import json
import base64
import codecs

# Assuming llm_module provides access to the LLM or we use openai client directly
# For simplicity, let's assume direct openai usage here, but adapt if using llm_module helpers
import openai
import os
from dotenv import load_dotenv

# Import parser for validation
from .parser_module import parse_code, load_languages # Need languages if not passed explicitly

# Load API Key (ensure this setup is appropriate for your project structure)
load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    # Handle missing key appropriately - maybe raise an error or log warning
    logging.error("OPENAI_API_KEY not found in environment variables.")
    # Consider fallback or raising an exception if key is essential
# else: # Optional: Set it if your openai library version requires it globally
#    openai.api_key = os.getenv("OPENAI_API_KEY")


logger = logging.getLogger(__name__)

def run_stage2_llm_strings(code: str, lang: str, languages: dict, **kwargs) -> str | None:
    """
    Performs Stage 2: String Deobfuscation using an LLM.

    Identifies common string obfuscations (like char arrays, base64, hex)
    and replaces them with plain string literals. Validates the output syntax.

    Args:
        code: The code string input (likely from Stage 1).
        lang: The programming language identifier.
        languages: Dictionary of loaded tree-sitter languages for validation.
        **kwargs: Additional keyword arguments (e.g., llm_model, temperature).

    Returns:
        The code string with deobfuscated strings, or the original code string
        if deobfuscation fails or produces invalid syntax. Returns None on critical failure.
    """
    logger.info(f"Starting Stage 2: String Deobfuscation for language '{lang}'...")

    # --- Define common string obfuscation patterns the LLM should look for ---
    # These are examples; the LLM should use its broader knowledge too.
    common_patterns_desc = """
    - Character Arrays/Lists: Initialized with integer char codes (e.g., `char s[] = {104, 101, ...};` in C, `let s = [104, 101, ...]` in JS/Rust, `s = [104, 101, ...]` in Python).
    - Hex Arrays/Strings: Similar to above but using hex values (e.g., `0x68, 0x65, ...`).
    - Base64 Encoded Strings: Often passed to a `decode` or `base64` function (e.g., `base64.b64decode('aGVsbG8=')`).
    - Concatenation of Small Strings/Chars: Building strings piece by piece (e.g., `'h' + 'e' + 'l' + 'l' + 'o'`).
    - Simple XOR/Shift Encryption: Strings encoded with a simple key and then decoded (e.g., looping through chars and XORing with a key). Look for decoding loops.
    - Custom Decoding Functions: Calls to functions with names like `decode_string`, `get_hidden_text`, `decrypt_data` that likely return a string.
    - Escaped Sequences: Overuse of octal or hex escapes within string literals.
    """

    # --- Construct the LLM Prompt ---
    prompt = f"""
You are an expert code analysis assistant specializing in deobfuscation for the '{lang}' language.
Your task is to identify and reverse common string obfuscation techniques within the provided code snippet.

Analyze the following '{lang}' code:
{code}


Identify instances where string literals appear to be obfuscated using techniques such as (but not limited to):
{common_patterns_desc}

Your goal is to replace these obfuscated representations *in place* with their original, plain string literal equivalents, formatted correctly for the '{lang}' language (e.g., ensure proper quoting and escaping within the final string literal).

**Instructions:**
1.  Carefully analyze the code to find obfuscated strings.
2.  Decode or reassemble the original string value.
3.  Modify the code to replace the obfuscated form with the plain string literal. For example, replace `char s[] = {{104, 101, 108, 108, 111}};` with `char* s = "hello";` (adjusting for '{lang}' syntax) or replace `print(base64.b64decode('aGVsbG8=').decode())` with `print("hello")`.
4.  **Crucially: Only modify the string representations.** Do NOT change variable names, logic, control flow, comments, or other parts of the code unless it's part of the string decoding mechanism itself (e.g., removing a decoding function call if its result is now inline).
5.  If you encounter a complex or unknown obfuscation method you cannot confidently reverse, **leave it unchanged**. Prioritize correctness.
6.  Return the **entire modified code block** as a single response. Do not add explanations, apologies, or markdown formatting around the code. Just the raw code.

Modified Code Only:
"""

    try:
        logger.info("Sending code to LLM for string deobfuscation...")
        # Use appropriate model and parameters
        llm_model = kwargs.get("llm_model", "gpt-4o-mini") # Or your preferred model
        temperature = kwargs.get("temperature", 0.2) # Low temp for predictable transformation

        # --- Make the LLM Call ---
        # Adjust based on your actual LLM client library (openai shown)
        response = openai.ChatCompletion.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": f"You are a code deobfuscation assistant for {lang}. You only output modified code."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max(2048, len(code.split()) + 512), # Ensure enough tokens for output
            n=1,
            stop=None
        )

        transformed_code = response.choices[0].message.content.strip()

        # --- Clean LLM Output ---
        # Remove potential markdown fences sometimes added by models
        transformed_code = re.sub(r'^```[a-zA-Z]*\s*', '', transformed_code)
        transformed_code = re.sub(r'\s*```$', '', transformed_code).strip()

        if not transformed_code:
            logger.warning("LLM returned empty response for string deobfuscation. Returning original code.")
            return code
        elif transformed_code == code:
             logger.info("LLM analysis found no strings to deobfuscate or made no changes.")
             # Return original code to signify no change happened that needs validation
             return code

        logger.info("LLM transformation complete. Validating syntax...")

        # --- Validate Syntax of Transformed Code ---
        try:
            parse_code(transformed_code, lang, languages)
            logger.info("Syntax validation successful for LLM-modified code.")
            return transformed_code # Return the validated, transformed code
        except Exception as parse_error:
            logger.error("LLM output failed syntax validation for language '%s'. Discarding changes.", lang)
            logger.error("Parser error: %s", parse_error)
            # Optionally log the failed code snippet for debugging:
            # logger.debug("Failed LLM Output:\n%s", transformed_code)
            return code # Return the *original* code if validation fails

    except Exception as e:
        logger.error(f"Error during Stage 2 LLM call or processing: {e}", exc_info=True)
        return code # Return original code on error

# Example usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Stage 2: String Deobfuscation")

    # Example 1: C code with char array
    c_code_obfuscated = """
#include <stdio.h>
int main() {
    char msg[] = {72, 101, 108, 108, 111, 44, 32, 87, 111, 114, 108, 100, 33, 0}; // "Hello, World!"
    printf("%s\\n", msg);
    return 0;
}
"""
    # Example 2: Python code with base64
    python_code_obfuscated = """
import base64

def show_secret():
    hidden = 'UHl0aG9uIGlzIGZ1bg==' # "Python is fun"
    decoded = base64.b64decode(hidden).decode('utf-8')
    print(decoded)

show_secret()
"""
    # Example 3: JS with concatenation
    js_code_obfuscated = """
function greet() {
    let part1 = "Java";
    let part2 = "Script";
    let message = part1 + part2 + " " + 'R' + 'o' + 'c' + 'k' + 's';
    console.log(message);
}
greet();
"""
    test_cases = [
        {"lang": "c", "code": c_code_obfuscated},
        {"lang": "python", "code": python_code_obfuscated},
        {"lang": "javascript", "code": js_code_obfuscated},
    ]

    try:
        loaded_langs = load_languages()
        logger.info("Languages loaded for testing.")
    except Exception as e:
        logger.error("Failed to load languages for test: %s", e)
        sys.exit(1)


    for i, test in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ({test['lang']}) ---")
        print("Obfuscated Code:")
        print(test["code"])
        print("-" * 20)
        result = run_stage2_llm_strings(test["code"], test["lang"], loaded_langs)
        print("Deobfuscated Code:")
        if result:
            print(result)
        else:
            print("Deobfuscation failed or returned None.")
        print("--- End Test Case ---\n")