# stage5_llm_deadcode.py

import logging

# Import the new LLM helper function and parser components
from .llm_module import remove_dead_code_llm
from .parser_module import parse_code

logger = logging.getLogger(__name__)

def run_stage5_llm_deadcode(code: str, lang: str, languages: dict, **kwargs) -> str | None:
    """
    Performs Stage 5: Dead Code Removal (Cautious LLM Pass).

    Orchestrates the process by calling the LLM module to attempt dead code
    removal and then validates the syntax of the result. Emphasizes that this
    stage is inherently risky and relies on LLM caution.

    Args:
        code: The code string input (likely from Stage 4).
        lang: The programming language identifier.
        languages: Dictionary of loaded tree-sitter languages for validation.
        **kwargs: Additional keyword arguments passed to the LLM module.

    Returns:
        The code string potentially with dead code removed, if successful and valid.
        Returns the original code string if removal fails, makes no changes,
        or produces invalid syntax. Returns None only on critical LLM failure.
    """
    logger.info(f"--- Starting Stage 5: Dead Code Removal (LLM Pass - Cautious) for language '{lang}' ---")
    logger.warning("Stage 5 (Dead Code Removal) relies on LLM interpretation and carries risk of removing necessary code. Review output carefully.")

    # Call the LLM function to perform the removal
    transformed_code = remove_dead_code_llm(code, lang, **kwargs)

    # Handle critical LLM failure
    if transformed_code is None:
        logger.error("Dead code removal failed due to LLM error. Returning code from previous stage.")
        return code

    # Check if code was actually changed
    if transformed_code == code:
        logger.info("LLM analysis found no dead code to remove or made no changes.")
        return code

    logger.info("LLM transformation complete. Validating syntax...")

    # Validate Syntax of Transformed Code
    try:
        parse_code(transformed_code, lang, languages)
        logger.info("Syntax validation successful for LLM-modified code.")
        return transformed_code # Return the validated, transformed code
    except Exception as parse_error:
        logger.error("LLM output failed syntax validation for language '%s'. Discarding Stage 5 changes.", lang)
        logger.error("Parser error: %s", parse_error)
        # logger.debug("Failed LLM Output:\n%s", transformed_code)
        return code # Return the *original* code if validation fails

# Example usage (for testing purposes)
if __name__ == '__main__':
    # Add necessary imports for testing context
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.parser_module import load_languages
    from dotenv import load_dotenv
    import os
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Stage 5: Dead Code Removal (Refactored)")

    # Example test cases
    test_code_py = """
import os

UNUSED_GLOBAL_VAR = 10

def used_func(x):
    print("Used:", x)
    return x + 1

def _unused_helper(y): # Conventionally private, potentially removable
    z = y * y # z is unused locally
    print("This might be dead")
    return y + UNUSED_GLOBAL_VAR # Accesses global

def main_logic(a):
    b = a * 2 # b is used
    c = 5     # c is unused
    d = "hello" # d is used
    if a > 10:
        used_func(b)
        print(d)
        return # Code after this in block is dead
        print("This is dead code")
    else:
        used_func(a)
    print("End logic") # This is reachable if a <= 10
    # _unused_helper is never called

main_logic(5)
main_logic(15)
"""
    # C example with unreachable and unused var
    test_code_c = """
#include <stdio.h>

static int helper_func(int factor) { // Static, potentially unused
    return factor * factor;
}

int main() {
    int i = 0;
    int limit = 10;
    int unused_var = 55; // Unused
    int result = 0;

    if (limit < 5) {
        return 1; // Unreachable below
        printf("This is unreachable\\n");
    }

    for (i = 0; i < limit; i++) {
        result += i;
        if (result > 100) {
             // Maybe call helper_func here if needed for testing its removal
             break; // Code after break in loop is reachable in next iteration
        }
        continue; // Code after continue in loop is unreachable in this iteration
        printf("Loop dead zone\\n");
    }
    printf("Result: %d\\n", result); // result is used
    // helper_func is never called
    // unused_var is never read
    return 0;
}
"""
    test_cases = [
        {"lang": "python", "code": test_code_py},
        {"lang": "c", "code": test_code_c},
    ]

    try:
        loaded_langs = load_languages()
        logger.info("Languages loaded for testing.")
    except Exception as e:
        logger.error("Failed to load languages for test: %s", e)
        sys.exit(1)

    if os.getenv("OPENAI_API_KEY"):
        for i, test in enumerate(test_cases):
            print(f"\n--- Test Case {i+1} ({test['lang']}) ---")
            print("Original Code:")
            print(test["code"])
            print("-" * 20)
            # Pass loaded languages
            result = run_stage5_llm_deadcode(test["code"], test["lang"], loaded_langs)
            print("Potentially Cleaned Code:")
            if result is None: print("Stage 5 failed critically.")
            elif result == test["code"]: print("Stage 5 made no changes or failed validation."); print(result)
            else: print(result)
            print("--- End Test Case ---\n")
    else:
        logger.warning("OPENAI_API_KEY not set. Skipping actual Stage 5 test run.")