# self_healing_code/app/ga_module.py

import random
import re
import logging # Added for better logging
from deap import base, creator, tools
import json # Added for potential LLM interaction

# Import mapping-based functions from the LLM module.
# Assumes llm_module now potentially has a function for suggesting improved names
from .llm_module import get_variable_mapping, apply_mapping_to_code, suggest_better_name # <-- Added suggest_better_name
# For AST validation and parsing.
from .parser_module import parse_code

logger = logging.getLogger(__name__) # Added logger

#####################################
# 1. Individual and Population Setup
#####################################
# (No changes needed here)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMax)

def init_individual(base_code: str, var_map: dict):
    """
    Initialize an individual containing:
    - 'base_code': the original code (never changes)
    - 'var_map': a mapping dict (original_identifier -> new_identifier)
    """
    ind = creator.Individual()
    ind["base_code"] = base_code
    ind["var_map"] = var_map
    return ind

def init_population(n: int, base_code: str, initial_mapping: dict):
    """
    Creates a population of individuals. For simplicity, every individual starts
    with the same base code and a copy of the initial mapping.
    """
    # Add slight variations to initial mappings to encourage diversity from the start
    population = []
    for i in range(n):
        new_mapping = dict(initial_mapping)
        # Optional: Introduce minor variations or use slightly different LLM prompts for initial diversity
        # For now, just creating copies
        population.append(init_individual(base_code, new_mapping))
    return population


#####################################
# 2. Rendering and AST Validation
#####################################
# (No changes needed here, relies on llm_module and parser_module)
def render_code(individual, lang: str, languages: dict):
    """
    Applies the individual's identifier mapping to the base code using the LLM,
    then parses the result with Tree-sitter to ensure syntactic validity.
    Returns the rendered code and its AST tree.
    """
    base_code = individual["base_code"]
    var_map = individual["var_map"]
    try:
        # Ensure mapping values are strings
        safe_map = {k: str(v) for k, v in var_map.items()}
        rendered_code = apply_mapping_to_code(base_code, safe_map, lang)
        tree = parse_code(rendered_code, lang, languages)
        # Handle potential None return from parse_code
        if tree is None:
             logger.warning(f"Code parsing failed for language {lang} after applying mapping. Code:\n{rendered_code[:500]}...")
             raise ValueError("Failed to parse rendered code.")
        return rendered_code, tree
    except Exception as e:
        logger.error(f"Error during code rendering or parsing: {e}")
        # Return original code and None tree to indicate failure? Or raise?
        # Raising allows the evaluation to catch it and assign low fitness.
        raise e # Re-raise to be caught by evaluate_individual

def normalize_sexp(sexp: str) -> str:
    """
    Normalizes an S-expression by stripping numeric location information and extra whitespace.
    """
    # (No changes needed here)
    normalized = re.sub(r":[0-9]+", "", sexp)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

#####################################
# 3. Fitness Evaluation
#####################################
def evaluate_individual(individual, lang: str, languages: dict, initial_map: dict): # Added initial_map
    """
    Evaluates an individual's fitness based on language-agnostic criteria:
    1. Semantic appropriateness of identifier names (vars & functions)
    2. Consistency in naming conventions
    3. Descriptiveness vs. length balance
    4. Context-awareness in naming
    5. Programming best practices
    6. Syntactic Validity (Implicitly checked by render_code)
    7. Improvement over initial map (Slight bonus for meaningful changes)

    This function extracts identifier information from the rendered code and combines multiple metrics.
    """
    try:
        rendered_code, tree = render_code(individual, lang, languages) # Tree might be useful later
        if tree is None: # Check if parsing failed in render_code
             return (0,) # Assign zero fitness if code is syntactically invalid

        # Extract identifier names and their contexts from the rendered code.
        # Consider using AST (tree) for more robust extraction if regex proves insufficient
        identifier_info = extract_identifier_info(rendered_code, lang) # Renamed function
        if not identifier_info:
            # Penalize if no identifiers found, maybe code is trivial or extraction failed
            return (0.1,) # Small non-zero fitness

        # Calculate various metrics for the identifier names.
        metrics = calculate_naming_metrics(identifier_info, individual["base_code"], individual["var_map"], initial_map) # Pass maps

        # Combine metrics into a final score (Adjusted weights)
        score = (
            metrics["semantics_score"] * 0.30 +
            metrics["consistency_score"] * 0.20 + # Slightly less weight
            metrics["descriptiveness_score"] * 0.25 + # More weight
            metrics["context_score"] * 0.15 +
            metrics["best_practices_score"] * 0.05 + # Less weight
            metrics["improvement_score"] * 0.05 # Added small bonus for improvement
        )

        # Ensure score is within bounds
        score = max(0.0, min(1.0, score))

        return (score,)
    except Exception as e:
        logger.error(f"Evaluation error for individual: {e}")
        return (0,) # Penalize individuals that cause errors during evaluation


#####################################
# 4. Identifier Info Extraction and Metrics (Enhanced)
#####################################

# Patterns need to be general enough for many languages
# These are indicative and might need refinement based on testing across languages
# Using Tree-sitter queries here would be far more robust but adds complexity
IDENTIFIER_PATTERNS = {
    # Variables: let x = ..., var y = ..., const z: type = ..., int count = ... (Python, JS, C-likes, Rust, Swift, etc.)
    "variable_assignment": r'(?:let|var|const|int|float|double|string|char|bool|auto|final|val)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]',
    # Function parameters: function(a, b), def func(param: type), void process(int input)
    "function_parameter": r'(?:def|function|func|fn|void|int|float|string|bool|auto|public|private|static|suspend)\s+\w+\s*\(([^)]*)\)',
    # Function definitions: def name(...), function name(...), ReturnType name(...)
    "function_definition": r'(?:def|function|func|fn|class|struct|enum)\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\(',
    # Class/Struct definitions: class Name { ... }
    "class_definition": r'(?:class|struct|enum|interface|trait)\s+([A-Z_][a-zA-Z0-9_]+)',
    # For loop variables: for (int i = ...), for item in items:
    "loop_variable": r'for\s*\(\s*(?:let|var|int|auto)?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[=:]',
    "loop_iterator": r'for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+', # Python/Rust style
}

# Keywords to ignore (expand this list based on target languages)
RESERVED_KEYWORDS = set(['if', 'else', 'for', 'while', 'return', 'class', 'struct', 'enum', 'interface',
                       'public', 'private', 'static', 'const', 'let', 'var', 'def', 'function', 'fn', 'func',
                       'int', 'float', 'string', 'bool', 'void', 'auto', 'new', 'delete', 'try', 'catch',
                       'import', 'export', 'from', 'package', 'namespace', 'use', 'true', 'false', 'null', 'nil',
                       'self', 'this', 'super', 'async', 'await', 'yield', 'match', 'case', 'switch', 'break',
                       'continue', 'pass', 'in', 'is', 'not', 'and', 'or', 'type', 'typeof', 'instanceof'])


def extract_identifier_info(code: str, lang: str) -> dict:
    """
    Language-agnostic approach to extract identifier names (variables, functions, classes) and contexts.
    Uses regex patterns, acknowledging limitations. AST traversal would be superior.
    """
    identifier_info = {}

    def add_identifier(name, context, expression=None):
        if name and name not in RESERVED_KEYWORDS and not name.isdigit() and name not in identifier_info:
             identifier_info[name] = {
                 "context": context,
                 "expressions": [expression.strip()] if expression else [],
                 "usages": [] # Populated later
             }
        elif name in identifier_info and expression:
             # Append expression if identifier already found
             if 'expressions' not in identifier_info[name]: identifier_info[name]['expressions'] = []
             identifier_info[name]['expressions'].append(expression.strip())


    # Pattern 1: Variable assignments
    assignments = re.findall(IDENTIFIER_PATTERNS["variable_assignment"], code)
    for var in assignments:
        # Need context around the match to get the expression, regex alone is hard
        # This part remains weak without AST or more complex regex
        add_identifier(var, "variable_assignment", "unknown_expression") # Placeholder

    # Pattern 2: Function parameters
    func_param_defs = re.findall(IDENTIFIER_PATTERNS["function_parameter"], code)
    for params_str in func_param_defs:
        params = params_str.split(',')
        for param in params:
            param = param.strip()
            if not param: continue
            # Try to extract name (e.g., "int count", "count: int", "count")
            match = re.search(r'(?:[\w\:]+\s+)?([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*[:=\[\]].*)?$', param)
            if match:
                param_name = match.group(1)
                add_identifier(param_name, "parameter")

    # Pattern 3: Function definitions
    func_defs = re.findall(IDENTIFIER_PATTERNS["function_definition"], code)
    for func_name in func_defs:
        add_identifier(func_name, "function_definition")

    # Pattern 4: Class definitions
    class_defs = re.findall(IDENTIFIER_PATTERNS["class_definition"], code)
    for class_name in class_defs:
         add_identifier(class_name, "class_definition")

    # Pattern 5: Loop variables (basic C-style)
    loop_vars = re.findall(IDENTIFIER_PATTERNS["loop_variable"], code)
    for var in loop_vars:
        add_identifier(var, "loop_variable")

    # Pattern 6: Loop iterators (Python/Rust style)
    loop_iters = re.findall(IDENTIFIER_PATTERNS["loop_iterator"], code)
    for var in loop_iters:
        add_identifier(var, "loop_iterator")


    # Find general identifier usages throughout the code.
    # This is difficult with regex; it finds *potential* usages.
    # Using \b (word boundary) helps avoid matching substrings within other words.
    all_potential_identifiers = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', code))
    found_identifiers = list(identifier_info.keys())

    for ident in found_identifiers:
        # Avoid re-calculating usages if already done (though this function recalculates all)
        identifier_info[ident]["usages"] = find_identifier_usages(code, ident)

    # Add identifiers found only through usage (less context, but still useful)
    # for potential_ident in all_potential_identifiers:
    #     if potential_ident not in identifier_info and potential_ident not in RESERVED_KEYWORDS and not potential_ident.isdigit() and len(potential_ident) > 1:
    #          # Could add these with 'unknown' context, but might add noise.
    #          # Let's rely on definition patterns for now.
    #          pass

    logger.debug(f"Extracted identifiers: {list(identifier_info.keys())}")
    return identifier_info

def find_identifier_usages(code: str, identifier: str) -> list:
    """
    Identifies potential identifier usages across the code using regex.
    Returns a list of dictionaries describing potential usage types.
    NOTE: This is heuristic and can misinterpret context. AST is better.
    """
    usages = []
    # Use word boundaries (\b) to match whole words only
    ident_pattern = r'\b' + re.escape(identifier) + r'\b'

    # Simple context checks using surrounding characters/keywords
    # These are very basic examples

    # Assignment (LHS): identifier = ...
    if re.search(rf'{ident_pattern}\s*=[^=]', code): # Check for single '='
        usages.append({"type": "assignment_lhs"})

    # Assignment (RHS): ... = identifier
    if re.search(rf'[^=]=\s*{ident_pattern}', code): # Check for single '='
         usages.append({"type": "assignment_rhs"})

    # Arithmetic: identifier +-*/ ... OR ... +-*/ identifier
    if re.search(rf'{ident_pattern}\s*[\+\-\*\/]', code) or re.search(rf'[\+\-\*\/]\s*{ident_pattern}', code):
        usages.append({"type": "arithmetic"})

    # Comparison: identifier == != < > ... OR ... == != < > identifier
    if re.search(rf'{ident_pattern}\s*(?:==|!=|<=|>=|<|>|===|!==)', code) or re.search(rf'(?:==|!=|<=|>=|<|>|===|!==)\s*{ident_pattern}', code):
        usages.append({"type": "comparison"})

    # Function call (as argument): func(..., identifier, ...)
    if re.search(rf'\w+\s*\([^)]*{ident_pattern}[^)]*\)', code):
         # Exclude the case where the identifier *is* the function being called
         if not re.search(rf'\b{identifier}\s*\(', code):
              usages.append({"type": "function_argument"})

    # Function call (being called): identifier(...)
    if re.search(rf'{ident_pattern}\s*\(', code):
        usages.append({"type": "function_call"})

    # Return statement: return identifier; / return identifier\n
    if re.search(rf'return\s+{ident_pattern}', code):
        usages.append({"type": "return"})

    # Conditional: if (... identifier ...)
    if re.search(rf'(?:if|while|elif|case)\s*\([^)]*{ident_pattern}[^)]*\)', code):
        usages.append({"type": "conditional_expression"})

    # Property/Method Access (LHS): identifier.property / identifier.method()
    if re.search(rf'{ident_pattern}\.\w+', code):
        usages.append({"type": "property_method_access_lhs"})

    # Property/Method Access (RHS): object.identifier
    if re.search(rf'\w+\.{ident_pattern}', code):
        usages.append({"type": "property_method_access_rhs"})

    # Array/Collection Access: identifier[...]
    if re.search(rf'{ident_pattern}\s*\[', code):
        usages.append({"type": "collection_access"})

    return usages


def calculate_naming_metrics(identifier_info: dict, original_code: str, current_map: dict, initial_map: dict) -> dict:
    """
    Calculates various naming metrics based on identifier info. Now considers functions.
    Includes a metric for improvement over the initial mapping.
    """
    metrics = {
        "semantics_score": 0.0,
        "consistency_score": 0.0,
        "descriptiveness_score": 0.0,
        "context_score": 0.0,
        "best_practices_score": 0.0,
        "improvement_score": 0.0 # New metric
    }
    if not identifier_info:
        return metrics

    identifier_names = list(identifier_info.keys())
    if not identifier_names:
        return metrics

    num_identifiers = len(identifier_names)

    # --- Metric Calculations (Keep existing logic, potentially refine patterns/scores) ---
    semantic_scores = []
    common_patterns = get_common_naming_patterns() # Reuse existing patterns
    for ident, info in identifier_info.items():
        ident_score = 0.5 # Base score
        context = info.get("context", "unknown")

        # Apply common patterns
        for pattern, modifier in common_patterns.items():
            if re.search(pattern, ident, re.IGNORECASE):
                 ident_score += modifier

        # Penalties/Bonuses based on length and context
        if len(ident) < 3 and ident.lower() not in ['i', 'j', 'k', 'x', 'y', 'z', 'id']: # Allow short common ones
             ident_score -= 0.3
        elif len(ident) > 25: # Penalize overly long names
             ident_score -= 0.2

        # Function specific checks
        if context == "function_definition":
            # Bonus for verb-based names (crude check)
            if any(ident.lower().startswith(verb) for verb in ["get", "set", "calculate", "process", "handle", "is", "has", "validate", "parse", "load", "save", "create", "update", "delete"]):
                 ident_score += 0.2
            # Penalize noun-like function names unless they are constructors/factories (hard to tell)
            elif not any(ident.lower().endswith(verb) for verb in ["er", "or", "tion", "ment"]): # Very basic check
                 pass # Avoid penalizing simple nouns for now

        # Class specific checks
        elif context == "class_definition":
             if not re.match(r'^[A-Z]', ident): # Penalize if not starting with Uppercase
                 ident_score -= 0.2
             if not any(ident.lower().endswith(suffix) for suffix in ["manager", "controller", "service", "provider", "handler", "util", "config", "model", "view"]):
                 pass # Noun-based is generally good here

        # Anti-pattern check
        if "mutated" in ident.lower(): # Keep penalty for "mutated" remnant
            ident_score -= 0.4 * ident.lower().count("mutated")
        if re.match(r'^(var|temp|tmp|data)\d*$', ident.lower()): # Penalize generic names
             ident_score -= 0.3

        semantic_scores.append(max(0, min(1.0, ident_score)))

    metrics["semantics_score"] = sum(semantic_scores) / num_identifiers if semantic_scores else 0

    # --- Consistency Score ---
    metrics["consistency_score"] = calculate_naming_consistency(identifier_names) # Use existing function

    # --- Descriptiveness Score --- (Combine with context)
    descriptive_context_scores = []
    for ident, info in identifier_info.items():
        context = info.get("context", "")
        usages = info.get("usages", [])
        score = 0.5 # Base

        # Context-based scoring (simplified examples)
        if context == "parameter":
            if len(ident) >= 3: score += 0.1
            if any(u.get("type") == "arithmetic" for u in usages):
                 if any(term in ident.lower() for term in ["num", "operand", "value", "factor"]): score += 0.2
            if any(u.get("type") == "conditional_expression" for u in usages):
                 if any(term in ident.lower() for term in ["flag", "status", "condition", "enable"]): score += 0.2

        elif context in ["loop_variable", "loop_iterator"]:
            if ident.lower() in ["i", "j", "k", "index", "idx", "counter"]: score += 0.3
            elif "index" in ident.lower() or "count" in ident.lower(): score += 0.2
            else: score -= 0.1 # Penalize non-standard loop vars slightly

        elif context == "function_definition":
             if len(ident) > 5: score += 0.1 # Encourage descriptive function names
             if any(u.get("type") == "return" for u in usages): # Does this make sense? Usage of the function name is 'call'
                 pass # Logic needs rethink - maybe check return value type if AST was used

        elif context == "class_definition":
             if len(ident) > 4: score += 0.2 # Encourage descriptive class names

        # Usage-based scoring
        if any(u.get("type") == "return" for u in usages):
            if any(term in ident.lower() for term in ["result", "output", "value", "data"]): score += 0.1
        if any(u.get("type") == "conditional_expression" for u in usages):
            if any(prefix in ident.lower() for prefix in ["is_", "has_", "should_", "can_"]) or ident.startswith("is") or ident.startswith("has"): score += 0.2
            elif any(term in ident.lower() for term in ["flag", "state", "status", "valid"]): score += 0.1

        # Combine descriptiveness and context scores
        descriptive_context_scores.append(max(0, min(1.0, score)))

    # Average them (crude combination)
    avg_desc_ctx_score = sum(descriptive_context_scores) / num_identifiers if descriptive_context_scores else 0
    metrics["descriptiveness_score"] = avg_desc_ctx_score
    metrics["context_score"] = avg_desc_ctx_score # Use the same combined score for now


    # --- Best Practices Score ---
    bp_scores = []
    for ident in identifier_names:
        bp_score = 0.5
        if len(ident) == 1 and ident.lower() not in ['i', 'j', 'k', 'x', 'y', 'z', 'n', 'm', 'c', 'e', 'f', 'g']: # Allow more single letters if common
             bp_score -= 0.2
        # Penalize names like temp1, data2 etc. unless specific domain (e.g. utf8)
        if re.match(r'^[a-zA-Z]+[0-9]+$', ident) and not re.match(r'^(utf|ascii|iso|md|sha)\d+$', ident.lower()):
            bp_score -= 0.2
        if ident.lower() in ["temp", "tmp", "var", "value", "val", "foo", "bar", "data", "obj", "o", "myvar"]: # More generics
            bp_score -= 0.2
        if 3 <= len(ident) <= 20: # Optimal length range
            bp_score += 0.1
        elif len(ident) > 25: # Penalize long names
            bp_score -= 0.2

        # Discourage remnants of mutation markers if they slip through
        if "mutated" in ident.lower() or "temp" in ident.lower() or "temp" in ident.lower():
             bp_score -= 0.3

        bp_scores.append(max(0, min(1.0, bp_score)))
    metrics["best_practices_score"] = sum(bp_scores) / num_identifiers if bp_scores else 0


    # --- Improvement Score ---
    # Calculate how many identifiers have changed *meaningfully* from the initial map
    changed_count = 0
    meaningful_change_count = 0
    current_ident_map = {}
    # Map original names to current names
    for orig_name, current_name in current_map.items():
         # Need to find which original name maps to the identifiers present *in the rendered code*
         # This is tricky because the LLM might rename things not in the initial map,
         # or the extraction might find identifiers not mapped.
         # Let's compare current names to initial names for *shared original keys*
         if orig_name in initial_map:
              initial_name = initial_map[orig_name]
              if current_name != initial_name:
                  changed_count += 1
                  # Define "meaningful": not just adding underscores or changing case slightly, or removing 'Mutated'
                  if initial_name.lower().replace("_", "") != current_name.lower().replace("_", "") and \
                     score_variable_name(current_name) > score_variable_name(initial_name) + 0.1: # Check if score improved
                      meaningful_change_count += 1

    total_mapped = len(initial_map)
    if total_mapped > 0:
         # Reward meaningful changes more
         metrics["improvement_score"] = (0.5 * (changed_count / total_mapped) + 0.5 * (meaningful_change_count / total_mapped))
    else:
         metrics["improvement_score"] = 0.0 # No initial map to compare against

    return metrics


def calculate_naming_consistency(identifier_names: list) -> float:
    """
    Evaluates naming consistency based on dominant casing style.
    Improved robustness for mixed lists or undefined styles.
    """
    if not identifier_names or len(identifier_names) < 2:
        return 1.0

    casing_patterns = {
        # Order matters slightly: check specific before general
        "PascalCase": r'^[A-Z][a-zA-Z0-9]*$', # Must start Upper
        "camelCase": r'^[a-z][a-zA-Z0-9]*$', # Must start lower
        "snake_case": r'^[a-z][a-z0-9_]*[a-z0-9]$', # Must start/end lower, allows underscore
        "UPPER_SNAKE": r'^[A-Z][A-Z0-9_]*[A-Z0-9]$', # Must start/end upper, allows underscore
        "kebab-case": r'^[a-z][a-z0-9\-]*[a-z0-9]$', # Allows hyphen
         # Add more if needed (e.g., Hungarian notation - harder to detect reliably)
    }
    style_counts = {style: 0 for style in casing_patterns}
    unmatched_count = 0

    for ident in identifier_names:
         # Ignore single-letter identifiers for consistency checks as they often don't conform
         if len(ident) <= 1:
              continue
         matched = False
         for style, pattern in casing_patterns.items():
             if re.match(pattern, ident):
                 # Prioritize PascalCase/UPPER_SNAKE if it also matches camel/snake
                 if style == "camelCase" and re.match(casing_patterns["PascalCase"], ident):
                      continue # Skip camel if it's Pascal
                 if style == "snake_case" and re.match(casing_patterns["UPPER_SNAKE"], ident):
                      continue # Skip snake if it's UPPER_SNAKE

                 style_counts[style] += 1
                 matched = True
                 break # Count first match (after priority checks)
         if not matched:
             unmatched_count += 1

    total_considered = len([ident for ident in identifier_names if len(ident)>1])
    if total_considered == 0: return 1.0 # All names were single letters

    if not style_counts or all(v == 0 for v in style_counts.values()):
         # If no patterns matched (e.g., all names are weird like 'a_B_c')
         return 0.0 # Low consistency

    # Find the count of the most frequent style
    dominant_count = max(style_counts.values())

    # Calculate consistency score
    consistency = dominant_count / total_considered
    return consistency

def get_common_naming_patterns() -> dict:
    """
    Returns a dictionary of common identifier naming patterns and associated score modifiers.
    Added patterns for functions/classes.
    """
    return {
        # General Concepts
        r'count|total|sum|average|mean': 0.2,
        r'index|idx|pos|offset': 0.2,
        r'result|output|return': 0.2,
        r'input|arg|param|data': 0.1, # Slightly less positive for generic input terms
        r'first|second|third|last|prev|next': 0.2,
        r'start|end|begin|finish|init': 0.2,
        r'min|max|limit|bound': 0.2,
        r'value|val': 0.1, # Generic
        r'name|key|id|identifier': 0.2,
        r'length|size|width|height|dim': 0.2,
        r'element|item|member|entry': 0.2,
        # Boolean/Flags
        r'^(?:is|has|can|should|allow|enable|check|validate)': 0.3,
        r'flag|status|state|valid|active|ready': 0.2,
        # Arithmetic/Numeric
        r'operand|factor|term|coeff': 0.3,
        r'multiplier|divisor|dividend|numerator|denominator|ratio': 0.3,
        # Functions/Actions (often verbs)
        r'get|set|add|remove|update|delete|create|find|search|load|save': 0.2,
        r'process|handle|execute|run|start|stop|parse|convert|render': 0.2,
        r'calculate|compute|measure|estimate': 0.2,
        # Collections
        r'list|array|vector|sequence|items|elements': 0.2,
        r'map|dict|table|lookup|cache': 0.2,
        r'queue|stack|deque': 0.2,
        r'set|group|collection': 0.2,
        # OO/Structure
        r'manager|controller|service|provider|repository|factory|builder': 0.2,
        r'handler|listener|observer|subscriber|interceptor': 0.2,
        r'config|settings|options|context|env': 0.2,
        r'util|helper|support|common': 0.1,
        r'model|view|presenter|entity|dto|record|struct': 0.2,
        r'node|edge|vertex|graph|tree': 0.2,
        r'root|parent|child|leaf': 0.2,
        # Penalties
        r'temp|tmp': -0.2,
        r'foo|bar|baz': -0.3,
        r'data|obj|info|detail': -0.1, # Penalize overly generic nouns
        r'^[a-rt-wz]$': -0.1, # Single letters (except common i,j,k,x,y,z)
        r'mutated': -0.4, # Stronger penalty
        r'^(var|val|temp|tmp|data)[0-9]+$': -0.3, # Penalize var1, data2 etc.
        r'^[a-z]{1,2}[0-9]+$': -0.2 # Penalize ab1, c2 etc.
    }

#####################################
# 5. Mutation Operator (LLM Enhanced)
#####################################

def mutate_mapping_llm(individual, lang: str, languages: dict):
    """
    LLM-enhanced, language-agnostic mutation operator.
    Selects one or more identifiers (variables or functions) from the *current* mapping,
    gets context-aware naming suggestions from an LLM, and updates the mapping.
    """
    current_map = individual["var_map"]
    if not current_map:
        logger.warning("Mutation skipped: Individual has empty mapping.")
        return individual, # No change if map is empty

    base_code = individual["base_code"] # Needed for context

    # Render the *current* code to analyze its state before mutation
    try:
        current_code, _ = render_code(individual, lang, languages)
        # Extract info from the *current* code to find candidate identifiers
        identifier_info = extract_identifier_info(current_code, lang)
    except Exception as e:
        logger.error(f"Mutation skipped: Error rendering or extracting info for mutation: {e}")
        return individual, # Skip mutation if rendering/extraction fails

    # Find original names corresponding to current identifiers
    # This requires reversing the map logic slightly or finding a better way
    # Let's try to mutate based on *original* names present in the map keys

    candidates = list(current_map.keys())
    if not candidates:
         logger.warning("Mutation skipped: No original names found in mapping keys.")
         return individual, # No keys to mutate

    # --- Strategy: Mutate 1 to 3 identifiers per call ---
    num_to_mutate = random.randint(1, min(3, len(candidates)))
    keys_to_mutate = random.sample(candidates, num_to_mutate)

    logger.debug(f"Attempting to mutate original keys: {keys_to_mutate}")

    successful_mutations = 0
    for original_key in keys_to_mutate:
        current_name = current_map.get(original_key)
        if not current_name:
            logger.warning(f"Skipping mutation for key '{original_key}': Not found in current map.")
            continue

        # Try to get context for the *current_name* in the *current_code*
        context_snippet = find_context_snippet(current_code, current_name) # Helper function needed

        try:
            # *** LLM Call for Suggestion ***
            suggested_name = suggest_better_name(
                original_name=original_key,
                current_name=current_name,
                code_context=context_snippet or current_code, # Provide snippet or full code
                lang=lang
            )

            if suggested_name and suggested_name != current_name:
                # Basic sanity check on suggested name
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_\-]*$', suggested_name) and suggested_name not in RESERVED_KEYWORDS:
                    logger.info(f"Mutating '{original_key}': '{current_name}' -> '{suggested_name}'")
                    current_map[original_key] = suggested_name # Update the individual's map
                    successful_mutations += 1
                else:
                    logger.warning(f"LLM suggested invalid name '{suggested_name}' for '{current_name}', skipping.")
            # else: No suggestion or same name returned by LLM

        except Exception as e:
            logger.error(f"Error getting LLM suggestion for '{current_name}': {e}")
            # Optionally, fallback to a simpler mutation? For now, just log and continue.

    # If mutation occurred, the fitness must be re-evaluated
    if successful_mutations > 0:
         del individual.fitness.values # Invalidate fitness

    return individual, # Return the possibly modified individual

def find_context_snippet(code: str, identifier: str, window=5) -> str:
    """
    Helper to find a few lines of code surrounding the first occurrence of an identifier.
    """
    lines = code.splitlines()
    ident_pattern = r'\b' + re.escape(identifier) + r'\b'
    first_occurrence_line = -1

    for i, line in enumerate(lines):
        if re.search(ident_pattern, line):
            first_occurrence_line = i
            break

    if first_occurrence_line != -1:
        start = max(0, first_occurrence_line - window)
        end = min(len(lines), first_occurrence_line + window + 1)
        return "\n".join(lines[start:end])
    else:
        return None # Identifier not found (shouldn't happen if called correctly)


# Remove or comment out the old rule-based mutation functions
# def mutate_mapping(...): ...
# def generate_improved_name(...): ...
# def determine_variable_role(...): ...
# def generate_role_based_name(...): ...


def score_variable_name(name: str) -> float:
    """
    Provides a *quick* score for an identifier name's quality.
    Used primarily in crossover, less detailed than full evaluation.
    Kept simple for speed.
    """
    score = 0.5
    if not name or not isinstance(name, str): return 0.0 # Handle potential non-string values

    name_lower = name.lower()

    # Penalties for obvious anti-patterns
    if "mutated" in name_lower: score -= 0.4 * name_lower.count("mutated")
    if name_lower in ["temp", "tmp", "var", "val", "foo", "bar", "baz", "data", "obj", "o", "myvar"]: score -= 0.2
    if re.match(r'^[a-zA-Z]+[0-9]+$', name) and not re.match(r'^(utf|ascii|iso|md|sha)\d+$', name_lower): score -= 0.15
    if len(name) == 1 and name_lower not in ['i', 'j', 'k', 'x', 'y', 'z', 'n', 'm', 'c', 'e', 'f', 'g']: score -= 0.2

    # Bonuses for length and potential keywords
    if 3 <= len(name) <= 20: score += 0.1
    elif len(name) > 25: score -= 0.15
    else: score -= 0.05 # Penalty for very short (2) or slightly long (21-25)

    common_terms = ["count", "total", "sum", "index", "result", "input", "output", "name", "key", "id", "value",
                    "list", "map", "array", "set", "queue", "stack", "node", "item", "element", "flag", "status",
                    "is", "has", "can", "should", "get", "set", "add", "process", "handle", "config", "util"]
    if any(term in name_lower for term in common_terms):
        score += 0.1

    if re.match(r'^[A-Z]', name) and not name.isupper(): # PascalCase likely
         score += 0.05
    elif re.match(r'^[a-z]', name) and '_' in name: # snake_case likely
         score += 0.05
    elif re.match(r'^[a-z]', name) and not '_' in name: # camelCase likely
         score += 0.05

    return max(0.0, min(1.0, score))


#####################################
# 6. Improved Crossover Operator
#####################################
# (Keep the existing improved_crossover_mapping, it's reasonable)
# It uses score_variable_name for efficiency.
def improved_crossover_mapping(ind1, ind2):
    """
    Intelligent crossover merging mappings based on quick quality scores (score_variable_name).
    Assigns the better-scoring name for shared keys to both offspring, with some randomness.
    """
    map1 = ind1["var_map"]
    map2 = ind2["var_map"]
    new_map1 = {}
    new_map2 = {}
    all_vars = set(map1.keys()) | set(map2.keys()) # Union of original variable keys

    for var_key in all_vars:
        name1 = map1.get(var_key)
        name2 = map2.get(var_key)

        # Ensure names are valid strings before scoring
        name1_str = str(name1) if name1 is not None else ""
        name2_str = str(name2) if name2 is not None else ""

        score1 = score_variable_name(name1_str) if name1_str else 0
        score2 = score_variable_name(name2_str) if name2_str else 0

        # Decide which name to propagate
        if name1_str and name2_str: # Both parents have a mapping for this key
            # Use a threshold to avoid swapping very similar scores, adds stability
            if score1 > score2 + 0.1:
                chosen_name = name1_str
            elif score2 > score1 + 0.1:
                chosen_name = name2_str
            else:
                # Scores are close, choose randomly or based on slight edge
                chosen_name = name1_str if score1 >= score2 else name2_str
                # Add small chance to keep original parent's name even if slightly worse
                if random.random() < 0.2:
                     new_map1[var_key] = name1_str
                     new_map2[var_key] = name2_str
                     continue # Skip common assignment below

            new_map1[var_key] = chosen_name
            new_map2[var_key] = chosen_name

        elif name1_str: # Only parent 1 has this key
            new_map1[var_key] = name1_str
            new_map2[var_key] = name1_str # Propagate to parent 2
        elif name2_str: # Only parent 2 has this key
            new_map1[var_key] = name2_str # Propagate to parent 1
            new_map2[var_key] = name2_str
        # else: Neither parent had the key (shouldn't happen with set union logic)

    ind1["var_map"] = new_map1
    ind2["var_map"] = new_map2
    # Invalidate fitness after crossover
    del ind1.fitness.values
    del ind2.fitness.values
    return ind1, ind2


#####################################
# 7. GA Setup and Execution (Adjusted)
#####################################
def run_ga(initial_code: str, lang: str, languages: dict, population_size: int = 20, generations: int = 15, initial_map_retries=2): # Increased defaults
    """
    Runs the mapping-based genetic algorithm:
    1. Obtains an initial identifier mapping (vars & funcs) from LLM. Includes retries.
    2. Initializes a population using the base code and the initial mapping.
    3. Evolves the population using LLM-enhanced mutation and quality-based crossover.
    4. Evaluation uses multiple naming metrics, including an improvement score.
    5. Uses an adaptive mutation rate.
    6. Returns the rendered (deobfuscated) code from the best individual found.
    """
    initial_map = None
    for attempt in range(initial_map_retries):
        try:
            logger.info(f"Attempting to get initial mapping (attempt {attempt+1}/{initial_map_retries})...")
            # *** Ensure get_variable_mapping asks for functions too ***
            initial_map = get_variable_mapping(initial_code, lang)
            if initial_map: # Break if successful
                 # Sanitize map: Ensure values are strings
                 initial_map = {str(k): str(v) for k, v in initial_map.items() if v} # Filter empty values
                 break
        except Exception as e:
            logger.error(f"Could not obtain initial mapping (attempt {attempt+1}): {e}")
            if attempt == initial_map_retries - 1:
                 logger.error("Failed to get initial mapping after multiple retries. Aborting.")
                 return initial_code # Return original code on failure
            # Optional: wait before retry? time.sleep(1)

    if not initial_map:
         logger.warning("Proceeding without an initial LLM mapping. GA might be less effective.")
         initial_map = {} # Start with an empty map if LLM fails completely

    logger.info(f"Initial Mapping ({len(initial_map)} identifiers): {initial_map}")

    pop = init_population(population_size, initial_code, initial_map)

    toolbox = base.Toolbox()
    # Pass initial_map to evaluator
    toolbox.register("evaluate", evaluate_individual, lang=lang, languages=languages, initial_map=initial_map)
    toolbox.register("mate", improved_crossover_mapping)
    # Use the new LLM-based mutation
    toolbox.register("mutate", mutate_mapping_llm, lang=lang, languages=languages)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Evaluate initial population.
    logger.info("Evaluating initial population...")
    try:
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            logger.debug(f"Initial fitness: {fit[0]:.4f}")
    except Exception as e:
        logger.error(f"Error evaluating initial population: {e}. Cannot proceed.")
        return initial_code

    # Adaptive mutation rate initial setting.
    base_mutation_prob = 0.3 # Slightly higher base mutation
    mutation_prob = base_mutation_prob
    stagnation_counter = 0
    max_stagnation = 4 # Increase mutation more aggressively if stagnated for this many gens

    hof = tools.HallOfFame(1) # Hall of Fame to store the best individual found

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda f: sum(v[0] for v in f) / len(f)) # Use f[0] for single objective
    stats.register("std", lambda f: (sum((v[0] - (sum(x[0] for x in f) / len(f)))**2 for v in f) / len(f))**0.5)
    stats.register("min", lambda f: min(v[0] for v in f))
    stats.register("max", lambda f: max(v[0] for v in f))

    logger.info("Starting GA evolution...")
    for gen in range(generations):
        # Selection
        offspring = toolbox.select(pop, len(pop))
        # Clone offspring to avoid modifying originals directly during crossover/mutation
        offspring = [creator.Individual(ind) for ind in offspring] # Use constructor for deep copy like behavior for the dict structure.

        # Apply Crossover
        # Iterate over pairs (doesn't modify the list size)
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.6: # Crossover probability
                toolbox.mate(offspring[i], offspring[i+1])
                # Fitness is invalidated inside mate function

        # Apply Mutation (Adaptive)
        mutated_count = 0
        for i in range(len(offspring)):
             if random.random() < mutation_prob:
                  toolbox.mutate(offspring[i])
                  mutated_count += 1
                  # Fitness is invalidated inside mutate function

        # Evaluate individuals with invalid fitness (those that underwent mate or mutate)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
             fitnesses = list(map(toolbox.evaluate, invalid_ind))
             for ind, fit in zip(invalid_ind, fitnesses):
                 ind.fitness.values = fit

        # Update Hall of Fame
        hof.update(offspring)

        # Replace population with offspring
        pop[:] = offspring

        # Gather statistics
        record = stats.compile(pop)
        best_current_score = record['max']
        best_overall_score = hof[0].fitness.values[0] # Best ever found

        logger.info(f"Gen {gen+1}/{generations}: "
                    f"Best Score: {best_current_score:.4f} (Overall: {best_overall_score:.4f}), "
                    f"Avg: {record['avg']:.4f}, Min: {record['min']:.4f}, "
                    f"Mutated: {mutated_count}, MutProb: {mutation_prob:.2f}")


        # Adaptive Mutation Rate Adjustment
        if hof.items[0].fitness.values[0] <= best_overall_score and gen > 0: # Compare with previous best overall
             stagnation_counter += 1
             logger.debug(f"Stagnation detected (counter: {stagnation_counter}). Best score: {best_overall_score:.4f}")
             if stagnation_counter >= max_stagnation // 2: # Start increasing earlier
                 mutation_prob = min(0.7, mutation_prob + 0.05) # Increase mutation rate more gradually
                 logger.debug(f"Increasing mutation probability to {mutation_prob:.2f}")

        else:
             # Improvement or first gen
             stagnation_counter = 0
             mutation_prob = base_mutation_prob # Reset to base if improvement occurs
             logger.debug(f"Improvement detected or first gen. Resetting mutation probability to {mutation_prob:.2f}")
             # Update best overall score if needed (already handled by HOF logic)


    # --- GA finished ---
    best_ind = hof[0] # Get the best individual from Hall of Fame
    logger.info(f"GA finished. Best overall score: {best_ind.fitness.values[0]:.4f}")
    logger.info(f"Best mapping found: {best_ind['var_map']}")

    try:
        # Render the final code from the best individual
        final_code, _ = render_code(best_ind, lang, languages)
        logger.info("Final code rendered successfully.")
        return final_code
    except Exception as e:
         logger.error(f"Failed to render final code from best individual: {e}")
         return initial_code # Fallback to initial code if final rendering fails

#####################################
# 8. Main / Example Usage
#####################################
if __name__ == "__main__":
    import sys
    sys.path.append('..') # Add parent directory to path if running script directly
    from app.parser_module import load_languages # Adjust import path if needed

    # More complex example
    sample_code_py = """
def proc_dat(d_list, thresh):
    res = []
    s = 0
    for i, x in enumerate(d_list):
        tmp = x * (i + 1)
        if tmp > thresh:
            s += tmp
            res.append(s)
        elif x < 0:
             res.append(x)
    return res, s
class My_Calc:
     def __init__(self, i_val):
          self.val_ = i_val
     def add_v(self, v2):
          self.val_ += v2
          return self.val_
"""

    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        langs = load_languages()
        if "python" not in langs:
            print("Error: Python language grammar not loaded.", file=sys.stderr)
            sys.exit(1)

        print("Running GA for Python code...")
        best_code = run_ga(
            initial_code=sample_code_py,
            lang="python",
            languages=langs,
            population_size=10, # Smaller for quick testing
            generations=5      # Fewer for quick testing
        )

        print("\n--- Initial Obfuscated Code ---")
        print(sample_code_py)
        print("\n--- Final Deobfuscated Code ---")
        if best_code:
            print(best_code)
        else:
            print("GA execution failed to produce final code.")

    except ImportError as e:
         print(f"Import Error: {e}. Make sure modules are correctly structured and dependencies are installed.", file=sys.stderr)
    except Exception as e:
         print(f"An unexpected error occurred: {e}", file=sys.stderr)
         import traceback
         traceback.print_exc()