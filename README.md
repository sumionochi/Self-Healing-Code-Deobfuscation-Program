self_healing_code/
├── app/
│   ├── __init__.py
│   ├── parser_module.py        # Loads Tree-sitter parsers and parses code
│   ├── llm_module.py           # Provides LLM-based code transformation functions
│   ├── ast_transformer.py      # Python-specific AST transformation
│   ├── integrated_transformer.py # Integrates LLM and AST-based transformation for multi-language support
│   ├── ga_module.py            # Implements the genetic algorithm for iterative deobfuscation
│   └── main.py                 # CLI entry point tying all components together
├── tests/                      # (Optional) Directory for unit/integration tests
├── build_languages.py          # (Optional) Script to build your Tree-sitter shared library (my-languages.so)
├── .env                      # Contains your OPENAI_API_KEY and other environment variables
└── requirements.txt            # Lists required packages (openai, python-dotenv, deap, tree-sitter, etc.)

Usage Example
To run in direct mode (using the integrated transformer):

bash
Copy
python -m app.main --codefile obfuscated_code.py --lang python --mode direct --output deobfuscated_code.py
To run in GA mode:

bash
Copy
python -m app.main --codefile obfuscated_code.py --lang python --mode ga --pop_size 10 --generations 5 --output deobfuscated_code.py
This version of main.py integrates the integrated_transformer for direct transformation while retaining the GA option for iterative improvements. Let me know if you need further adjustments or explanations!

python build_languages.py
python app/parser_module.py

python -m venv venv
source venv/bin/activate