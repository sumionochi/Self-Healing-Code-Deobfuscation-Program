# stage4_llm_expressions.py

import logging

# Import the new LLM helper function and parser components
from .llm_module import simplify_expressions_llm
from .parser_module import parse_code

logger = logging.getLogger(__name__)

def run_stage4_llm_expressions(code: str, lang: str, languages: dict, **kwargs) -> str | None:
    """
    Performs Stage 4: Expression Simplification (Constants & Booleans).

    Orchestrates the process by calling the LLM module to perform the
    simplification and then validates the syntax of the result.

    Args:
        code: The code string input (likely from Stage 3).
        lang: The programming language identifier.
        languages: Dictionary of loaded tree-sitter languages for validation.
        **kwargs: Additional keyword arguments passed to the LLM module.

    Returns:
        The code string with potentially simplified expressions, if successful and valid.
        Returns the original code string if simplification fails, makes no changes,
        or produces invalid syntax. Returns None only on critical LLM failure.
    """
    logger.info(f"--- Starting Stage 4: Expression Simplification (LLM Pass) for language '{lang}' ---")

    # Call the LLM function to perform the simplification
    transformed_code = simplify_expressions_llm(code, lang, **kwargs)

    # Handle critical LLM failure
    if transformed_code is None:
        logger.error("Expression simplification failed due to LLM error. Returning code from previous stage.")
        return code # Return original code passed into this stage

    # Check if code was actually changed
    if transformed_code == code:
        logger.info("LLM analysis found no expressions to simplify or made no changes.")
        return code # Return original code if no changes occurred

    logger.info("LLM transformation complete. Validating syntax...")

    # Validate Syntax of Transformed Code
    try:
        parse_code(transformed_code, lang, languages)
        logger.info("Syntax validation successful for LLM-modified code.")
        return transformed_code # Return the validated, transformed code
    except Exception as parse_error:
        logger.error("LLM output failed syntax validation for language '%s'. Discarding Stage 4 changes.", lang)
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
    # Ensure dotenv is loaded if running directly and OPENAI_API_KEY is needed
    from dotenv import load_dotenv
    import os
    load_dotenv()

    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Stage 4: Expression Simplification (Refactored)")

    # Example test cases
    test_code_py = """
def complex_calc(a, b):
    c = 5 * 2 + 4 # Constant folding
    d = a * 1 + 0 # Arithmetic identity
    flag1 = True
    flag2 = False
    # Boolean simplification: !(flag1 == False and flag2 == True) -> !(False and False) -> !False -> True
    complex_bool = not (flag1 == False and flag2 == True)
    if complex_bool == True: # Trivial condition
        print("Condition met")
    return c * d
"""
    test_code_js = """
function process(x, y) {
    const MAX_RETRIES = 3 + 1; // Constant folding
    let offset = y - 0; // Arithmetic identity
    let isValid = true;
    let shouldProceed = !(isValid === false); // Boolean simplification -> isValid
    if (shouldProceed) {
        for(let i = 0; i < MAX_RETRIES; i++) {
            console.log(offset + i);
        }
    }
}
"""
    test_cases = [
        {"lang": "python", "code": test_code_py},
        {"lang": "javascript", "code": test_code_js},
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
            result = run_stage4_llm_expressions(test["code"], test["lang"], loaded_langs)
            print("Simplified Code:")
            if result is None: print("Simplification failed critically.")
            elif result == test["code"]: print("Simplification made no changes or failed validation."); print(result)
            else: print(result)
            print("--- End Test Case ---\n")
    else:
        logger.warning("OPENAI_API_KEY not set. Skipping actual Stage 4 test run.")