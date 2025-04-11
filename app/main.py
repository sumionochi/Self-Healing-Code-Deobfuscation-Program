import argparse
import logging
from pathlib import Path

from .integrated_transformer import transform_code_integrated
from .ga_module import run_ga
from .parser_module import load_languages, parse_code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Self-Healing Code: Multi-language deobfuscation using integrated LLM transformation with AST validation (direct or GA-based)."
    )
    parser.add_argument(
        "--codefile",
        type=Path,
        required=True,
        help="Path to the file containing the obfuscated code."
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=[
            "python", "javascript", "c", "php", "scala", "jsdoc", "css", "ql",
            "regex", "html", "java", "bash", "typescript", "julia", "haskell",
            "c_sharp", "embedded_template", "agda", "verilog", "toml", "swift"
        ],
        help="The programming language of the code."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "ga"],
        default="direct",
        help="Transformation mode: 'direct' uses the integrated transformer; 'ga' uses a genetic algorithm for iterative improvement."
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=10,
        help="Population size for GA (only used in 'ga' mode)."
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Number of generations for GA (only used in 'ga' mode)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file for the deobfuscated code. If not provided, output to console."
    )
    args = parser.parse_args()

    if not args.codefile.is_file():
        logger.error("The specified code file does not exist: %s", args.codefile)
        return

    try:
        with open(args.codefile, "r", encoding="utf-8") as f:
            code = f.read()
        logger.info("Loaded code from %s", args.codefile)
    except Exception as e:
        logger.error("Error reading code file: %s", e)
        return

    languages = load_languages()
    deobfuscated_code = None

    if args.mode == "direct":
        try:
            deobfuscated_code = transform_code_integrated(code, args.lang, languages)
            logger.info("Direct integrated transformation successful.")
        except Exception as e:
            logger.error("Direct integrated transformation failed: %s", e)
            return
    elif args.mode == "ga":
        try:
            # Validate initial code via Tree-sitter.
            _ = parse_code(code, args.lang, languages)
        except Exception as e:
            logger.warning("Initial code AST validation issue: %s", e)
        try:
            deobfuscated_code = run_ga(
                initial_code=code,
                population_size=args.pop_size,
                generations=args.generations,
                lang=args.lang,
                languages=languages
            )
            logger.info("GA-based transformation complete.")
        except Exception as e:
            logger.error("GA transformation failed: %s", e)
            return

    if deobfuscated_code is None:
        logger.error("No deobfuscated code produced.")
        return

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(deobfuscated_code)
            logger.info("Deobfuscated code written to %s", args.output)
        except Exception as e:
            logger.error("Failed to write output file: %s", e)
    else:
        print("\nDeobfuscated Code:\n")
        print(deobfuscated_code)

if __name__ == "__main__":
    main()
