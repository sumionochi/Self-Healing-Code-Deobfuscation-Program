# self_healing_code/app/integrated_transformer.py

import logging
from .llm_module import get_variable_mapping, rename_variables_multilang
from .parser_module import parse_code
from .ast_transformer import transform_code as ast_transform_code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transform_code_integrated(code: str, lang: str, languages: dict) -> str:
    """
    Transforms the given code by integrating LLM-based transformation with an AST-based approach.
    
    For Python:
      1. Uses get_variable_mapping() to obtain a mapping of obfuscated variable names.
      2. Applies the AST-based transformation (via ast_transform_code) to rename variables while preserving syntax.
    
    For non-Python languages:
      1. Uses rename_variables_multilang() to directly transform the code.
    
    In all cases, the transformed code is validated by re-parsing it with Tree-sitter.
    
    :param code: The original source code.
    :param lang: The language of the code (e.g., "python", "javascript", etc.).
    :param languages: Dictionary of loaded Tree-sitter language parsers.
    :return: The transformed code if validation succeeds.
    """
    transformed_code = None
    if lang.lower() == "python":
        try:
            # Obtain mapping from the LLM.
            mapping = get_variable_mapping(code, lang)
            logger.info("Obtained mapping for %s: %s", lang, mapping)
            # Apply the AST-based transformation to ensure proper structure.
            transformed_code = ast_transform_code(code, mapping)
            logger.info("AST-based transformation complete for Python.")
        except Exception as e:
            logger.error("Error during AST-based transformation: %s", e)
            raise e
    else:
        try:
            # For non-Python languages, directly transform using the LLM.
            transformed_code = rename_variables_multilang(code, lang)
            logger.info("Direct LLM transformation complete for %s.", lang)
        except Exception as e:
            logger.error("Error during direct transformation for %s: %s", lang, e)
            raise e

    # Validate the transformed code using Tree-sitter.
    try:
        _ = parse_code(transformed_code, lang, languages)
        logger.info("Syntax validation successful for transformed %s code.", lang)
    except Exception as e:
        logger.error("Syntax validation failed: %s", e)
        raise Exception("Transformed code failed syntax validation.")

    return transformed_code

# For local testing (optional)
if __name__ == "__main__":
    from .parser_module import load_languages
    sample_code = "def x(a, b):\n    return a + b"
    languages = load_languages()
    try:
        result = transform_code_integrated(sample_code, "python", languages)
        print("Transformed Code:\n", result)
    except Exception as err:
        print("Transformation failed:", err)
