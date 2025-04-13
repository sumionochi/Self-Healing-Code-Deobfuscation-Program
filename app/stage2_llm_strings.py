# stage2_llm_strings.py

import logging

# Import the new LLM helper function and parser components
from .llm_module import deobfuscate_strings_llm
from .parser_module import parse_code

logger = logging.getLogger(__name__)

def run_stage2_llm_strings(code: str, lang: str, languages: dict, **kwargs) -> str | None:
    """
    Performs Stage 2: String Deobfuscation.

    Orchestrates the process by calling the LLM module to perform the
    deobfuscation and then validates the syntax of the result.

    Args:
        code: The code string input (likely from Stage 1).
        lang: The programming language identifier.
        languages: Dictionary of loaded tree-sitter languages for validation.
        **kwargs: Additional keyword arguments passed to the LLM module (e.g., llm_model).

    Returns:
        The code string with potentially deobfuscated strings, if successful and valid.
        Returns the original code string if deobfuscation fails, makes no changes,
        or produces invalid syntax. Returns None only on critical LLM failure.
    """
    logger.info(f"--- Starting Stage 2: String Deobfuscation (LLM Pass) for language '{lang}' ---")

    # Call the LLM function to perform the deobfuscation
    transformed_code = deobfuscate_strings_llm(code, lang, **kwargs)

    # Handle critical LLM failure
    if transformed_code is None:
        logger.error("String deobfuscation failed due to LLM error. Returning original code.")
        return code # Return original code, as None signifies failure in main loop

    # Check if code was actually changed
    if transformed_code == code:
        logger.info("LLM analysis found no strings to deobfuscate or made no changes.")
        return code # Return original code if no changes occurred

    logger.info("LLM transformation complete. Validating syntax...")

    # Validate Syntax of Transformed Code
    try:
        parse_code(transformed_code, lang, languages)
        logger.info("Syntax validation successful for LLM-modified code.")
        return transformed_code # Return the validated, transformed code
    except Exception as parse_error:
        logger.error("LLM output failed syntax validation for language '%s'. Discarding Stage 2 changes.", lang)
        logger.error("Parser error: %s", parse_error)
        # logger.debug("Failed LLM Output:\n%s", transformed_code) # Optional: log bad code
        return code # Return the *original* code if validation fails

# Example usage (for testing purposes)
if __name__ == '__main__':
    # Add necessary imports for testing context
    import sys
    from pathlib import Path
    # This assumes you might run this file directly for testing
    # Adjust path if necessary based on your execution context
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.parser_module import load_languages # Example: Adjust import path

    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Stage 2: String Deobfuscation (Refactored)")

    # Example test cases (same as before)
    c_code_obfuscated = """
        #include <stdio.h>
        int main() {
            char msg[] = {72, 101, 108, 108, 111, 44, 32, 87, 111, 114, 108, 100, 33, 0};
            printf("%s\\n", msg);
            return 0;
        }
        """
    python_code_obfuscated = """
        import base64
        def show_secret():
            hidden = 'UHl0aG9uIGlzIGZ1bg=='
            decoded = base64.b64decode(hidden).decode('utf-8')
            print(decoded)
        show_secret()
        """
    js_code_obfuscated = """
        function greet() {
            let part1 = "Java"; let part2 = "Script";
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

    # Check if API key exists before trying to run tests that call LLM
    if os.getenv("OPENAI_API_KEY"):
        for i, test in enumerate(test_cases):
            print(f"\n--- Test Case {i+1} ({test['lang']}) ---")
            print("Obfuscated Code:")
            print(test["code"])
            print("-" * 20)
            # Pass loaded languages to the function
            result = run_stage2_llm_strings(test["code"], test["lang"], loaded_langs)
            print("Deobfuscated Code:")
            # Check if result is None or same as input
            if result is None:
                 print("Deobfuscation failed critically.")
            elif result == test["code"]:
                 print("Deobfuscation made no changes or failed validation.")
                 print(result) # Print the original code
            else:
                 print(result)
            print("--- End Test Case ---\n")
    else:
        logger.warning("OPENAI_API_KEY not set. Skipping actual Stage 2 test run.")