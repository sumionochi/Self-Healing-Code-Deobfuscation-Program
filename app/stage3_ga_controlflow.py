# stage3_ga_controlflow.py

import logging
import random
import re
from deap import base, creator, tools, algorithms # Import algorithms
import math # For infinity

# Import the new LLM helper and parser components
from .llm_module import simplify_control_flow_llm
from .parser_module import parse_code, load_languages # Need languages if not passed explicitly

logger = logging.getLogger(__name__)

# --- GA Configuration ---

# Define possible simplification flags the GA can control
# These names MUST match exactly what simplify_control_flow_llm expects
SIMPLIFICATION_FLAGS = [
    "simplify_constant_conditions", # e.g., if(true), while(false)
    "remove_redundant_else",       # e.g., else block after return/break/continue in if
    "unwrap_single_statement_blocks", # e.g., if (x) { y=1; } -> if (x) y=1; (language dependent)
    "simplify_redundant_jumps",    # e.g., goto label; label: ... -> ... (less common now)
    "merge_nested_ifs",            # e.g., if(a) { if(b) {...} } -> if(a && b) {...}
    "simplify_trivial_loops",      # e.g., for loops with 0 or 1 iteration if detectable
]

# Fitness aims to minimize complexity and length (lower is better)
# Weights: (Cyclomatic Complexity, Code Length) - Negative because we minimize
# Using infinity for penalty ensures invalid solutions are heavily penalized.
WORST_FITNESS = (float('inf'), float('inf'))
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -0.5))
# Individual holds a list of booleans (flags)
creator.create("IndividualStage3", list, fitness=creator.FitnessMin) # Use list for individuals

toolbox = base.Toolbox()

# Attribute generator: A boolean value (0 or 1)
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers: Create individuals as lists of booleans
toolbox.register("individual", tools.initRepeat, creator.IndividualStage3, toolbox.attr_bool, len(SIMPLIFICATION_FLAGS))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- Genetic Operators ---
toolbox.register("mate", tools.cxTwoPoint) # Standard crossover for lists
toolbox.register("mutate", tools.mutFlipBit, indpb=0.15) # Mutate by flipping bits (flags)
toolbox.register("select", tools.selTournament, tournsize=3) # Selection method

# --- Evaluation Helpers ---

def calculate_cyclomatic_complexity(code: str) -> int:
    """Approximates cyclomatic complexity by counting decision points."""
    count = 1
    count += len(re.findall(r'\b(if|while|for|case|catch|goto)\b', code))
    count += len(re.findall(r'(&&|\|\||\?\:)', code)) # C-style, JS, Java etc.
    count += len(re.findall(r'\b(and|or)\b', code)) # Python, etc.
    count += len(re.findall(r'\b(elif|else if)\b', code))
    return count

# --- Fitness Function ---

def evaluate_control_flow(individual: list, # Type hint: expecting a list of 0s/1s
                          base_code: str,
                          lang: str,
                          languages: dict,
                          **kwargs):
    """
    Fitness function for control flow simplification.
    1. Determine active flags from the individual.
    2. Call LLM module to apply simplifications based on flags.
    3. Validate syntax of the result.
    4. Calculate complexity metrics (lower is better).
    """
    # Create a list of active flag names based on the individual (list of 0/1)
    active_flags = [SIMPLIFICATION_FLAGS[i] for i, active in enumerate(individual) if active]

    if not active_flags:
        # No flags active, calculate complexity of base code
        base_complexity = calculate_cyclomatic_complexity(base_code)
        base_length = len(base_code)
        logger.debug("Individual has no active flags. Evaluating base code complexity.")
        # Return complexity/length directly as fitness values (lower is better)
        return base_complexity, base_length

    # Call the LLM function from llm_module
    simplified_code = simplify_control_flow_llm(base_code, lang, active_flags, **kwargs)

    if simplified_code is None:
        logger.warning("LLM call failed critically for control flow simplification.")
        return WORST_FITNESS # Return worst fitness on critical LLM failure

    if simplified_code == base_code:
        # LLM made no changes (might be because no simplifications were applicable/safe)
        base_complexity = calculate_cyclomatic_complexity(base_code)
        base_length = len(base_code)
        logger.debug("LLM made no changes. Evaluating base code complexity.")
        return base_complexity, base_length

    # Validate Syntax of the simplified code
    try:
        parse_code(simplified_code, lang, languages)
        # Syntax is valid, calculate metrics
        complexity = calculate_cyclomatic_complexity(simplified_code)
        length = len(simplified_code)
        logger.debug(f"Valid syntax. Complexity: {complexity}, Length: {length}")
        # Return fitness tuple (lower is better)
        return complexity, length
    except Exception as parse_error:
        logger.warning(f"LLM output failed syntax validation: {parse_error}")
        # Penalize invalid syntax heavily
        return WORST_FITNESS

# --- Main Stage Function ---

def run_stage3_ga_controlflow(initial_code: str, lang: str, languages: dict, population_size: int, generations: int, **kwargs) -> str | None:
    """
    Runs the GA for Stage 3: Control Flow Simplification.

    Args:
        initial_code: Code string from the previous stage.
        lang: Language identifier.
        languages: Loaded tree-sitter languages.
        population_size: Population size for the GA.
        generations: Number of generations for the GA.
        **kwargs: Additional parameters passed to evaluation (e.g., llm_model).

    Returns:
        The simplified code string from the best individual found, or the
        original code if the GA fails or finds no valid improvements.
        Returns None only on critical setup error before GA loop.
    """
    logger.info("--- Starting Stage 3: Control Flow Simplification (GA+LLM) ---")
    if not initial_code:
         logger.error("Initial code for Stage 3 is empty.")
         return None # Cannot proceed

    # Register the evaluation function with necessary arguments for this run
    # Important: Pass initial_code here, not globally, if it changes run-to-run
    toolbox.register("evaluate", evaluate_control_flow,
                     base_code=initial_code, lang=lang, languages=languages, **kwargs)

    # Create population
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1) # Store the best individual

    # Setup statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda f: tuple(sum(v[i] for v in f) / len(f) for i in range(len(f[0]))) if f else (0,0))
    stats.register("min", lambda f: tuple(min(v[i] for v in f) for i in range(len(f[0]))) if f else (0,0))
    stats.register("max", lambda f: tuple(max(v[i] for v in f) for i in range(len(f[0]))) if f else (0,0)) # Max = Worst

    # --- Run GA ---
    try:
        logger.info(f"Starting GA evolution for {generations} generations with population size {population_size}...")
        # Use a standard evolutionary algorithm
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.2, # Adjusted probabilities
                                           ngen=generations, stats=stats,
                                           halloffame=hof, verbose=True)
        logger.info("GA evolution completed.")
        logger.info(f"Logbook:\n{logbook}")

    except Exception as e:
        logger.error(f"Error during GA execution for Stage 3: {e}", exc_info=True)
        logger.warning("Stage 3 failed during GA run. Returning original code.")
        # Ensure original code is returned, not None, unless initial_code was None
        return initial_code if initial_code else None

    # --- Process Results ---
    if not hof:
        logger.warning("Hall of Fame is empty after GA run. No valid solution found?")
        return initial_code

    best_ind = hof[0]
    # Ensure fitness values exist and are valid before accessing
    if not best_ind.fitness.valid or best_ind.fitness.values == WORST_FITNESS:
         logger.warning("Best individual had invalid or worst possible fitness. No valid simplification found.")
         return initial_code

    best_fitness = best_ind.fitness.values
    logger.info(f"GA finished. Best individual fitness (Complexity, Length): {best_fitness[0]:.2f}, {best_fitness[1]:.2f}")

    # Get the flags from the best individual
    active_best_flags = [SIMPLIFICATION_FLAGS[i] for i, active in enumerate(best_ind) if active]
    logger.info(f"Best flag set found: {active_best_flags}")

    if not active_best_flags:
        logger.info("Best individual found had no active simplification flags. Returning original code.")
        return initial_code

    # --- Re-run LLM with the best flag set to get the final code ---
    logger.info("Rendering final code using the best flag set found by GA...")
    final_code = simplify_control_flow_llm(initial_code, lang, active_best_flags, **kwargs)

    if final_code is None: # Critical LLM failure on final render
        logger.error("Final LLM call failed for best individual. Returning original code.")
        return initial_code
    elif final_code == initial_code:
        logger.info("Final rendering with best flags resulted in no changes from original.")
        return initial_code

    # Final validation check
    try:
        parse_code(final_code, lang, languages)
        logger.info("Final code syntax validation successful.")
        return final_code
    except Exception as e:
        logger.error(f"Error validating final code syntax for best individual: {e}", exc_info=True)
        logger.warning("Final simplified code failed validation. Returning original code.")
        return initial_code # Fallback to original on final validation failure

# Example usage (for testing) - Requires API Key and built languages
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Stage 3: Control Flow Simplification (GA+LLM - Refactored)")

    test_code_c = """
#include <stdio.h>
int main() {
    int x = 10, y = 0, z = 5;
    if (1) { printf("Always printed.\\n"); } // Constant condition
    if (x > 5) { if (z == 5) { y = x + z; } } // Nested ifs
    else { /* Empty else */ }
    if (y > 10) { printf("Y > 10\\n"); return 0; } else { printf("Y <= 10\\n"); } // Else not redundant
    while(0) { printf("Never printed.\\n"); } // Constant condition loop
    printf("Final y: %d\\n", y); return 0;
}
"""
    try:
        # Assumes parser_module is accessible
        loaded_langs = load_languages()
        logger.info("Languages loaded for testing.")
        if os.getenv("OPENAI_API_KEY"):
             result = run_stage3_ga_controlflow(
                 initial_code=test_code_c,
                 lang="c",
                 languages=loaded_langs,
                 population_size=10, generations=3 # Small run for testing
             )
             print("\n--- Original Code ---")
             print(test_code_c)
             print("\n--- Potentially Simplified Code (Stage 3) ---")
             print(result if result else "Stage 3 execution failed or returned None.")
             print("--- End Test ---")
        else:
             logger.warning("OPENAI_API_KEY not set. Skipping actual Stage 3 test run.")
    except Exception as e:
        logger.error("Error during Stage 3 test setup or run: %s", e, exc_info=True)