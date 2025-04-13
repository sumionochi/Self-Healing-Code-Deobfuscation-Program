# stage6_llm_format.py

import logging

# Import the new LLM helper function and parser components
from .llm_module import format_comment_code_llm
from .parser_module import parse_code

logger = logging.getLogger(__name__)

def run_stage6_llm_format(code: str, lang: str, languages: dict, **kwargs) -> str | None:
    """
    Performs Stage 6: Formatting & Commenting (LLM Pass).

    Orchestrates the process by calling the LLM module to format the code
    and add comments/docstrings, then validates the syntax of the result.

    Args:
        code: The code string input (likely from Stage 5).
        lang: The programming language identifier.
        languages: Dictionary of loaded tree-sitter languages for validation.
        **kwargs: Additional keyword arguments passed to the LLM module.

    Returns:
        The formatted and commented code string, if successful and valid.
        Returns the original code string if the process fails, makes no changes,
        or produces invalid syntax. Returns None only on critical LLM failure.
    """
    logger.info(f"--- Starting Stage 6: Formatting & Commenting (LLM Pass) for language '{lang}' ---")

    # Call the LLM function to perform formatting and commenting
    transformed_code = format_comment_code_llm(code, lang, **kwargs)

    # Handle critical LLM failure
    if transformed_code is None:
        logger.error("Formatting/Commenting failed due to LLM error. Returning code from previous stage.")
        return code

    # Check if code was actually changed
    # Note: Formatting changes might only be whitespace, so direct string comparison
    # might not capture formatting-only changes effectively if whitespace isn't preserved perfectly.
    # However, if comments/docstrings are added, it should differ.
    if transformed_code.strip() == code.strip(): # Use strip to ignore leading/trailing whitespace diffs
        logger.info("LLM analysis made no significant formatting/commenting changes.")
        # Decide whether to return original or potentially slightly whitespace-modified version
        # Returning transformed_code is usually safe here.
        # Let's check syntax anyway, just in case.
        pass # Proceed to validation even if only whitespace might have changed

    logger.info("LLM transformation complete. Validating syntax...")

    # Validate Syntax of Transformed Code
    # Formatting or commenting *shouldn't* break syntax, but LLM errors are possible.
    try:
        parse_code(transformed_code, lang, languages)
        logger.info("Syntax validation successful for formatted/commented code.")
        return transformed_code # Return the validated, transformed code
    except Exception as parse_error:
        logger.error("LLM output failed syntax validation after formatting/commenting for language '%s'. Discarding Stage 6 changes.", lang)
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
    logger.info("Testing Stage 6: Formatting & Commenting (Refactored)")

    # Example Python code (could be output from previous stages)
    test_code_py = """
def calculate_average(numbers_list): # Function expects list
    # Check if list is not empty to avoid division by zero
    if not numbers_list: return 0.0 # Return 0 for empty list
    total_sum=sum(numbers_list) # Calculate sum using built-in
    # Divide sum by count to get average
    avg=total_sum/len(numbers_list)
    return avg # Return calculated average
"""
    # Example C code
    test_code_c = """
#include <stdio.h>
// Function to calculate sum; assumes arr is not NULL
int calculate_sum(int *arr, int size){ int s = 0; for(int i=0; i<size; ++i){ s+=arr[i];} return s;}
int main() { int data[] = {1,2,3,4,5}; int len = sizeof(data)/sizeof(data[0]); int total=calculate_sum(data, len); printf("Sum: %d\\n", total); return 0;}
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
            print("Input Code (Pre-Stage 6):")
            print(test["code"])
            print("-" * 20)
            # Pass loaded languages
            result = run_stage6_llm_format(test["code"], test["lang"], loaded_langs)
            print("Formatted & Commented Code:")
            if result is None: print("Stage 6 failed critically.")
            elif result == test["code"]: print("Stage 6 made no significant changes or failed validation."); print(result)
            else: print(result)
            print("--- End Test Case ---\n")
    else:
        logger.warning("OPENAI_API_KEY not set. Skipping actual Stage 6 test run.")