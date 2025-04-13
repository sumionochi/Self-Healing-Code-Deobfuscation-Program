import argparse
import logging
import sys
import difflib
from pathlib import Path
import shutil

# --- Stage Function Imports (Placeholders) ---
from .stage1_ga_rename import run_stage1_ga_rename
from .stage2_llm_strings import run_stage2_llm_strings
from .stage3_ga_controlflow import run_stage3_ga_controlflow
from .stage4_llm_expressions import run_stage4_llm_expressions
from .stage5_llm_deadcode import run_stage5_llm_deadcode
from .stage6_llm_format import run_stage6_llm_format

# --- Parser Import ---
from .parser_module import load_languages, parse_code # Added parse_code here

# Keep direct mode as an option? Maybe for baseline comparison.
from .integrated_transformer import transform_code_integrated

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions for Interactive Mode ---
def display_diff(text1, text2, filename=""):
    # (display_diff function remains the same as before)
    diff = difflib.unified_diff(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        fromfile=f"before_{filename}",
        tofile=f"after_{filename}",
        lineterm='',
    )
    print("\n--- Changes Made ---")
    output = "".join(diff)
    if not output.strip():
        print("No significant textual changes detected.")
    else:
        for line in output.splitlines():
            if line.startswith('+') and not line.startswith('+++'): print(f"\033[92m{line}\033[0m")
            elif line.startswith('-') and not line.startswith('---'): print(f"\033[91m{line}\033[0m")
            elif line.startswith('^'): print(f"\033[94m{line}\033[0m")
            elif line.startswith('@@'): print(f"\033[96m{line}\033[0m")
            else: print(line)
    print("--- End Changes ---\n")

def prompt_for_action(prompt_text):
    # (prompt_for_action function remains the same as before)
    valid_inputs = ['p', 'r', 'j', 'a', 'y', 'n'] + [str(i) for i in range(1, 11)] + ['b', 'e']
    while True:
        action = input(f"{prompt_text} ").strip().lower()
        if action in valid_inputs:
            return action
        else:
            print("Invalid input, please try again.")

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Self-Healing Code: Multi-stage deobfuscation using GA and LLM."
    )
    # --- Arguments ---
    parser.add_argument("--codefile", type=Path, required=True, help="Path to the input code file.")
    parser.add_argument("--lang", type=str, required=True, help="Programming language (e.g., python, c, java).")
    parser.add_argument("--mode", type=str, choices=["direct", "multi_stage"], default="multi_stage", help="Transformation mode.")
    parser.add_argument("--interactive", action='store_true', help="Enable interactive mode ('multi_stage' only).")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file for the final code.")
    parser.add_argument("--temp_dir", type=Path, default="./deobfuscation_temp", help="Directory for intermediate stage outputs.")
    parser.add_argument("--pop_size", type=int, default=15, help="Population size for GA stages (1 & 3).")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations for GA stages (1 & 3).")
    args = parser.parse_args()

    # --- Input Validation ---
    if not args.codefile.is_file():
        logger.error("Input code file not found: %s", args.codefile)
        return
    if args.interactive and args.mode != "multi_stage":
        logger.warning("--interactive flag requires --mode 'multi_stage'. Disabling interactive mode.")
        args.interactive = False

    # --- Setup: Load Code and Languages ---
    try:
        initial_code = args.codefile.read_text(encoding="utf-8")
        logger.info("Loaded code from %s", args.codefile)
    except Exception as e:
        logger.error("Error reading code file %s: %s", args.codefile, e)
        return

    try:
        languages = load_languages() # Load tree-sitter languages early
        logger.info("Loaded Tree-sitter languages.")
    except Exception as e:
         logger.error("Failed to load tree-sitter languages. Ensure grammar files are built/accessible: %s", e)
         return

    # --- Initial Code Syntax Validation ---
    try:
        logger.info("Validating initial code syntax using Tree-sitter for language: %s", args.lang)
        # Parse the initial code. We don't need the tree object itself,
        # just want to ensure it parses without raising an exception.
        parse_code(initial_code, args.lang, languages) # Use the imported function
        logger.info("Initial code syntax validation successful.")
    except Exception as e:
        logger.error("Initial code failed syntax validation for language '%s'. Cannot proceed.", args.lang)
        logger.error("Parser error: %s", e, exc_info=True) # Log traceback for debugging
        return # Exit if initial code is invalid

    # --- Proceed with Transformation ---
    final_code = None
    args.temp_dir.mkdir(parents=True, exist_ok=True) # Ensure temp dir exists

    # --- Mode Selection ---
    if args.mode == "direct":
        logger.info("Running in 'direct' mode...")
        try:
            final_code = transform_code_integrated(initial_code, args.lang, languages)
            logger.info("Direct integrated transformation successful.")
        except Exception as e:
            logger.error("Direct integrated transformation failed: %s", e, exc_info=True)
            return

    elif args.mode == "multi_stage":
        logger.info("Running in 'multi_stage' mode...")

        # Define intermediate file paths helper
        def get_stage_output_path(stage_num):
            return args.temp_dir / f"{args.codefile.stem}_stage{stage_num}{args.codefile.suffix}"

        # Stage Definitions (updated based on previous agreement)
        common_params = {"lang": args.lang, "languages": languages}
        ga_params = {"pop_size": args.pop_size, "generations": args.generations}
        stages = {
            1: {"name": "Renaming (GA+LLM)", "func": run_stage1_ga_rename, "is_ga": True},
            2: {"name": "String Deobfuscation (LLM)", "func": run_stage2_llm_strings, "is_ga": False},
            3: {"name": "Control Flow Simp. (GA+LLM)", "func": run_stage3_ga_controlflow, "is_ga": True},
            4: {"name": "Expression Simp. (LLM)", "func": run_stage4_llm_expressions, "is_ga": False},
            5: {"name": "Dead Code Removal (LLM)", "func": run_stage5_llm_deadcode, "is_ga": False},
            6: {"name": "Formatting/Commenting (LLM)", "func": run_stage6_llm_format, "is_ga": False}
        }

        # State tracking
        output_files = {0: args.codefile}
        for i in range(1, 7): output_files[i] = None
        last_successful_stage = 0
        current_code_content = initial_code

        # Stage Execution Helper (updated slightly for clarity)
        def run_stage(stage_num, input_code_str):
            if stage_num not in stages:
                logger.error("Invalid stage number: %d", stage_num)
                return None, None
            stage_info = stages[stage_num]
            output_path = get_stage_output_path(stage_num)
            logger.info("--- Starting Stage %d: %s ---", stage_num, stage_info["name"])

            params_for_stage = {**common_params}
            if stage_info["is_ga"]:
                params_for_stage.update(ga_params)
                # Ensure GA stages receive the code via 'initial_code' parameter
                params_for_stage['initial_code'] = input_code_str
            else:
                # Ensure non-GA stages receive code via 'code' parameter
                params_for_stage['code'] = input_code_str

            try:
                # Call the stage function with prepared parameters
                result_code = stage_info["func"](**params_for_stage)

                if result_code and isinstance(result_code, str):
                    output_path.write_text(result_code, encoding="utf-8")
                    logger.info("Stage %d completed successfully. Output: %s", stage_num, output_path)
                    return result_code, output_path
                elif result_code is None:
                     logger.error("Stage %d function returned None.", stage_num)
                     return None, None
                else:
                     logger.error("Stage %d function returned unexpected type: %s", stage_num, type(result_code))
                     return None, None
            except Exception as e:
                logger.error("Stage %d (%s) failed: %s", stage_num, stage_info["name"], e, exc_info=True)
                return None, None

        # --- Automated vs Interactive Execution ---
        if not args.interactive:
            # --- Automated Sequential Execution ---
            logger.info("Running automated multi-stage process...")
            for stage_num in range(1, 7): # Loop through stages 1 to 6
                result_code, output_path = run_stage(stage_num, current_code_content)
                if result_code is None:
                    logger.error("Automated process failed at Stage %d. Using output from previous stage.", stage_num)
                    final_code = current_code_content
                    break # Exit loop
                current_code_content = result_code
                output_files[stage_num] = output_path
                last_successful_stage = stage_num
            else: # Loop completed without break
                final_code = current_code_content
                logger.info("Automated multi-stage process completed.")

        else:
            # --- Interactive Menu Execution ---
            logger.info("Starting interactive multi-stage process...")
            # (Interactive loop logic remains the same as previous version)
            previous_code_content = initial_code
            while True:
                print("\n--- Interactive Deobfuscation Menu ---")
                print(f"Current State: Output of Stage {last_successful_stage} available.")
                current_input_display_path = output_files[last_successful_stage] or "Original File"
                print(f"Input for next stage will be: {current_input_display_path}")
                print("-" * 30)
                for i in range(1, 7):
                    status = " (Completed)" if output_files[i] else " (Pending)"
                    ga_tag = " (GA+LLM)" if stages[i]["is_ga"] else " (LLM)"
                    print(f"[{i}] Run Stage {i}{status}: {stages[i]['name']}{ga_tag}")
                print("-" * 30)
                print("[7] Run All Remaining Stages Sequentially")
                print("[8] View Diff from Previous Stage")
                print("[9] View Current Code")
                print("[10] Exit")

                choice = prompt_for_action("Choose option:")
                next_stage_to_run = -1

                if choice.isdigit() and 1 <= int(choice) <= 6:
                    next_stage_to_run = int(choice)
                    # Confirmation/Warning Logic...
                    if next_stage_to_run <= last_successful_stage:
                        if prompt_for_action(f"Stage {next_stage_to_run} already completed. Re-run? (y/n):") != 'y': continue
                    elif next_stage_to_run > last_successful_stage + 1:
                         if prompt_for_action(f"Warning: Run Stage {next_stage_to_run} before {last_successful_stage + 1}? (y/n):") != 'y': continue

                    # Determine input code...
                    input_stage_num = next_stage_to_run - 1
                    input_path = output_files[input_stage_num]
                    if not input_path or not input_path.is_file():
                        logger.error("Cannot run Stage %d: Input file from Stage %d (%s) not found or invalid.", next_stage_to_run, input_stage_num, input_path)
                        continue
                    try: input_code_str = input_path.read_text(encoding='utf-8')
                    except Exception as e: logger.error("Error reading input file %s: %s", input_path, e); continue

                    # Execute stage...
                    previous_code_content = input_code_str
                    result_code, output_path = run_stage(next_stage_to_run, input_code_str)

                    # Post-execution update...
                    if result_code is not None:
                        display_diff(previous_code_content, result_code, f"stage_{next_stage_to_run}")
                        output_files[next_stage_to_run] = output_path
                        last_successful_stage = max(last_successful_stage, next_stage_to_run)
                        current_code_content = result_code
                    else:
                        logger.error("Stage %d failed. State reset to output of Stage %d.", next_stage_to_run, last_successful_stage)
                        # Revert current_code_content? Let's keep it as the last successful for now
                        # current_code_content = previous_code_content # Option to revert fully

                elif choice == '7': # Run All Remaining
                    start_stage = last_successful_stage + 1
                    logger.info(f"Running remaining stages ({start_stage} to 6) sequentially...")
                    temp_current_code = current_code_content
                    success = True
                    if start_stage > 6: logger.info("All stages already completed."); continue
                    for stage_num in range(start_stage, 7):
                        input_code_str_for_run = temp_current_code
                        logger.info(f"Auto-running Stage {stage_num}...")
                        result_code, output_path = run_stage(stage_num, input_code_str_for_run)
                        if result_code is None: logger.error("Sequence failed at Stage %d.", stage_num); success = False; break
                        temp_current_code = result_code
                        output_files[stage_num] = output_path
                        last_successful_stage = stage_num
                        current_code_content = temp_current_code
                    if success: logger.info("Remaining stages executed successfully.")
                    final_code = current_code_content

                elif choice == '8': # View Diff
                    # (Diff logic remains same)
                    if last_successful_stage == 0: print("No previous stage output exists to compare with the original.")
                    elif not output_files[last_successful_stage]: print(f"Stage {last_successful_stage} did not complete successfully.")
                    else:
                        prev_stage_num = last_successful_stage - 1
                        prev_path = output_files[prev_stage_num]
                        curr_path = output_files[last_successful_stage]
                        try:
                            prev_code = initial_code if prev_stage_num == 0 else prev_path.read_text(encoding='utf-8')
                            curr_code = curr_path.read_text(encoding='utf-8')
                            display_diff(prev_code, curr_code, f"stage_{last_successful_stage}")
                        except Exception as e: logger.error("Could not read files for diff: %s", e)

                elif choice == '9': # View Current Code
                    print(f"\n--- Current Code (Output of Stage {last_successful_stage}) ---")
                    print(current_code_content)
                    print("--- End Current Code ---")

                elif choice == '10':
                    logger.info("Exiting interactive mode.")
                    final_code = current_code_content
                    break # Exit loop

                else:
                    print("Invalid choice.")

            if final_code is None: # If loop exited early without setting final code
                final_code = current_code_content


    # --- Output ---
    if final_code is None:
        logger.error("Processing failed or was aborted. No final deobfuscated code produced.")
        return

    if args.output:
        try:
            args.output.write_text(final_code, encoding="utf-8")
            logger.info("Final deobfuscated code written to %s", args.output)
        except Exception as e:
            logger.error("Failed to write output file %s: %s", args.output, e)
    else:
        print("\n--- Final Deobfuscated Code ---")
        print(final_code)
        print("--- End Final Code ---")

    logger.info("Note: Intermediate files (if any) are kept in %s", args.temp_dir)


if __name__ == "__main__":
    main()