# self_healing_code/app/parser_module.py

import os
from tree_sitter import Language, Parser

# Define the path to the shared library built earlier
LIB_PATH = os.path.join(os.getcwd(), "build", "my-languages.so")

def load_languages():
    """
    Loads and returns a dictionary of supported languages.
    Make sure the language names (the second parameter) match those defined in each grammar.
    """
    languages = {
        "python": Language(LIB_PATH, "python"),
        "javascript": Language(LIB_PATH, "javascript"),
        "c": Language(LIB_PATH, "c"),
        "php": Language(LIB_PATH, "php"),
        "scala": Language(LIB_PATH, "scala"),
        "jsdoc": Language(LIB_PATH, "jsdoc"),
        "css": Language(LIB_PATH, "css"),
        "ql": Language(LIB_PATH, "ql"),
        "regex": Language(LIB_PATH, "regex"),
        "html": Language(LIB_PATH, "html"),
        "java": Language(LIB_PATH, "java"),
        "bash": Language(LIB_PATH, "bash"),
        "typescript": Language(LIB_PATH, "typescript"),
        "julia": Language(LIB_PATH, "julia"),
        "haskell": Language(LIB_PATH, "haskell"),
        "c_sharp": Language(LIB_PATH, "c_sharp"),
        "embedded_template": Language(LIB_PATH, "embedded_template"),
        "agda": Language(LIB_PATH, "agda"),
        "verilog": Language(LIB_PATH, "verilog"),
        "toml": Language(LIB_PATH, "toml"),
        "swift": Language(LIB_PATH, "swift")
    }
    return languages

def parse_code(code: str, lang: str, languages: dict):
    """
    Parses the given code string into an AST using the specified language parser.
    
    Parameters:
      - code: The source code as a string.
      - lang: The language key (e.g., "python", "javascript", "c", "php", "scala", "jsdoc", "css", "ql", "regex", "html", "java", "bash", "typescript", "julia", "haskell", "c_sharp", "embedded_template", "agda", "verilog", "toml", "swift" etc.).
      - languages: A dictionary of loaded languages.
    
    Returns:
      - The parsed AST.
    """
    lang_key = lang.lower()
    if lang_key not in languages:
        raise ValueError(f"Unsupported language: {lang}")
    parser = Parser()
    parser.set_language(languages[lang_key])
    tree = parser.parse(code.encode())
    return tree

if __name__ == "__main__":
    # For testing: load languages and parse a sample snippet.
    langs = load_languages()
    sample_code = "def hello():\n    print('Hello World')"
    tree = parse_code(sample_code, "python", langs)
    print("AST Root Node:", tree.root_node.sexp())
