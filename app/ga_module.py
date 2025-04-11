# self_healing_code/app/ga_module.py

import random
import re
from deap import base, creator, tools
from .llm_module import rename_variables_multilang  # Multi-language LLM transformation
from .parser_module import parse_code  # For AST validation

# -------------------------------
# Define the fitness and individual
# -------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def init_individual(code: str):
    """
    Initialize an individual represented as a list containing a single code string.
    """
    return creator.Individual([code])

toolbox.register("individual", init_individual)
toolbox.register("population", lambda n, code: [init_individual(code) for _ in range(n)])

def normalize_sexp(sexp: str) -> str:
    """
    Normalizes an S-expression by removing numeric location information and extra whitespace.
    This helps compare AST structures in a more relaxed way.
    """
    # Remove numbers (and colons) that often represent position info.
    normalized = re.sub(r":[0-9]+", "", sexp)
    # Remove any extra whitespace.
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

# -------------------------------
# Mutation Operator with Structural Consistency Check and AST Validation
# -------------------------------
def mutation_operator_with_validation(individual, lang, languages):
    """
    Mutates an individual's code using the LLM-based renaming function (multi-language version)
    and then validates the output. It parses both the original and mutated code with Tree-sitter,
    normalizes their S-expressions, and only accepts the mutation if the normalized structure is identical.
    """
    original_code = individual[0]
    try:
        mutated_code = rename_variables_multilang(original_code, lang)
        # Validate by parsing both codes.
        mutated_tree = parse_code(mutated_code, lang, languages)
        original_tree = parse_code(original_code, lang, languages)
        
        # Normalize S-expressions to remove position data.
        orig_sexp = normalize_sexp(original_tree.root_node.sexp())
        mut_sexp = normalize_sexp(mutated_tree.root_node.sexp())
        
        if orig_sexp != mut_sexp:
            print("Mutation resulted in a structural change; discarding mutation.")
            return individual,
        
        individual[0] = mutated_code
    except Exception as e:
        print(f"Mutation validation error: {e}. Keeping original code.")
    return individual,

# -------------------------------
# Enhanced Evaluation Function (Fitness)
# -------------------------------
def evaluate_individual(individual):
    """
    Evaluate the readability of code using a composite heuristic:
      - Average length of variable names.
      - Uniqueness ratio: unique names / total names.
      - Fraction of short names (length < 3) as a penalty.
    Returns a tuple containing the final score.
    """
    code = individual[0]
    var_names = re.findall(r'\b[a-zA-Z_][a-zA-Z_0-9]*\b', code)
    if not var_names:
        return (0,)
    
    avg_length = sum(len(var) for var in var_names) / len(var_names)
    unique_count = len(set(var_names))
    uniqueness_ratio = unique_count / len(var_names)
    short_count = sum(1 for var in var_names if len(var) < 3)
    fraction_short = short_count / len(var_names)
    
    score = avg_length + 10 * uniqueness_ratio - 5 * fraction_short
    return (score,)

toolbox.register("evaluate", evaluate_individual)

# -------------------------------
# Crossover Operator
# -------------------------------
def crossover_operator(ind1, ind2):
    """
    Performs a simple crossover by splitting the code strings at the midpoint.
    """
    code1, code2 = ind1[0], ind2[0]
    mid1, mid2 = len(code1) // 2, len(code2) // 2
    new_code1 = code1[:mid1] + code2[mid2:]
    new_code2 = code2[:mid2] + code1[mid1:]
    ind1[0], ind2[0] = new_code1, new_code2
    return ind1, ind2

toolbox.register("mate", crossover_operator)

# -------------------------------
# Selection Operator
# -------------------------------
toolbox.register("select", tools.selTournament, tournsize=3)

# -------------------------------
# Run the Genetic Algorithm with AST Validation and Normalized Structural Consistency
# -------------------------------
def run_ga(initial_code: str, population_size: int = 10, generations: int = 5, lang: str = "python", languages: dict = None):
    """
    Run the genetic algorithm:
      - Initialize a population of individuals from the obfuscated code.
      - Evolve the population over several generations.
      - Use AST validation and normalized structural consistency check after each mutation.
      - Return the deobfuscated code from the best individual.
    """
    if languages is None:
        raise ValueError("Languages dictionary must be provided for AST validation.")
        
    pop = toolbox.population(population_size, initial_code)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Override the mutation operator to include our improved check.
    toolbox.register("mutate", lambda ind: mutation_operator_with_validation(ind, lang, languages))
    
    for gen in range(generations):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation (with AST and normalized structural validation)
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Re-evaluate individuals with invalid fitness values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        best = tools.selBest(pop, 1)[0]
        best_snippet = best[0][:100].replace("\n", " ")
        print(f"Generation {gen+1}: Best Score = {best.fitness.values[0]:.4f} | Snippet: {best_snippet}...")
    
    best_ind = tools.selBest(pop, 1)[0]
    return best_ind[0]

if __name__ == "__main__":
    from .parser_module import load_languages
    sample_code = "def x(a, b): return a+b"
    langs = load_languages()
    deobfuscated_code = run_ga(sample_code, population_size=10, generations=5, lang="python", languages=langs)
    print("Deobfuscated Code:")
    print(deobfuscated_code)
