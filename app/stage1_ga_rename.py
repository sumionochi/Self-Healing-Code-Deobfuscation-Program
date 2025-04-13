# self_healing_code/app/stage1_ga_rename.py

import random
import re
import logging
from deap import base, creator, tools
import json
import math # For potential calculations

# Import mapping-based functions from the LLM module.
# Assume llm_module.suggest_better_name now returns a LIST of suggestions
from .llm_module import get_variable_mapping, apply_mapping_to_code, suggest_better_names # Renamed for clarity
# For AST validation and parsing.
from .parser_module import parse_code

logger = logging.getLogger(__name__)

#####################################
# 1. Individual and Population Setup
#####################################
# Add 'history' to track changes? Could be complex. Stick with map for now.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMax)

# (init_individual and init_population remain the same)
def init_individual(base_code: str, var_map: dict):
    ind = creator.Individual()
    ind["base_code"] = base_code
    ind["var_map"] = var_map
    return ind

def init_population(n: int, base_code: str, initial_mapping: dict):
    return [init_individual(base_code, dict(initial_mapping)) for _ in range(n)]


#####################################
# 2. Rendering and AST Validation
#####################################
# (render_code and normalize_sexp remain the same)
def render_code(individual, lang: str, languages: dict):
    base_code = individual["base_code"]
    var_map = individual["var_map"]
    try:
        safe_map = {k: str(v) for k, v in var_map.items()}
        rendered_code = apply_mapping_to_code(base_code, safe_map, lang)
        tree = parse_code(rendered_code, lang, languages)
        if tree is None:
             logger.warning(f"Code parsing failed for language {lang} after applying mapping. Code:\n{rendered_code[:500]}...")
             raise ValueError("Failed to parse rendered code.")
        return rendered_code, tree
    except Exception as e:
        logger.error(f"Error during code rendering or parsing: {e}")
        raise e

def normalize_sexp(sexp: str) -> str:
    normalized = re.sub(r":[0-9]+", "", sexp)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

#####################################
# 3. Fitness Evaluation (Refined Weights & Logic)
#####################################
def evaluate_individual(individual, lang: str, languages: dict, initial_map: dict):
    """
    Evaluates an individual's fitness. Adjusted weights and logic
    to potentially reward significant improvements more.
    """
    try:
        rendered_code, tree = render_code(individual, lang, languages)
        if tree is None:
             return (0,)

        identifier_info = extract_identifier_info(rendered_code, lang)
        if not identifier_info:
            return (0.1,)

        # Calculate metrics
        metrics = calculate_naming_metrics(identifier_info, individual["base_code"], individual["var_map"], initial_map)

        # --- Adjusted Weights ---
        # Increase weight for semantics and descriptiveness, slightly reduce consistency,
        # increase weight for improvement score to encourage change.
        score = (
            metrics["semantics_score"] * 0.35 +        # Increased
            metrics["consistency_score"] * 0.15 +      # Decreased
            metrics["descriptiveness_score"] * 0.25 +  # Kept same (already high)
            metrics["context_score"] * 0.10 +          # Decreased (often overlaps descriptiveness)
            metrics["best_practices_score"] * 0.05 +   # Kept low
            metrics["improvement_score"] * 0.10        # Increased significantly
        )

        # Ensure score is within bounds
        score = max(0.0, min(1.0, score))

        return (score,)
    except Exception as e:
        logger.error(f"Evaluation error for individual: {e}")
        return (0,)


#####################################
# 4. Identifier Info Extraction and Metrics (Refined Scoring)
#####################################

# (IDENTIFIER_PATTERNS and RESERVED_KEYWORDS remain the same)
IDENTIFIER_PATTERNS = {
    "variable_assignment": r'(?:let|var|const|int|float|double|string|char|bool|auto|final|val)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]',
    "function_parameter": r'(?:def|function|func|fn|void|int|float|string|bool|auto|public|private|static|suspend)\s+\w+\s*\(([^)]*)\)',
    "function_definition": r'(?:def|function|func|fn|class|struct|enum)\s+([a-zA-Z_][a-zA-Z0-9_]+)\s*\(',
    "class_definition": r'(?:class|struct|enum|interface|trait)\s+([A-Z_][a-zA-Z0-9_]+)',
    "loop_variable": r'for\s*\(\s*(?:let|var|int|auto)?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[=:]',
    "loop_iterator": r'for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+in\s+',
}
RESERVED_KEYWORDS = set(['if', 'else', 'for', 'while', 'return', 'class', 'struct', 'enum', 'interface',
                       'public', 'private', 'static', 'const', 'let', 'var', 'def', 'function', 'fn', 'func',
                       'int', 'float', 'string', 'bool', 'void', 'auto', 'new', 'delete', 'try', 'catch',
                       'import', 'export', 'from', 'package', 'namespace', 'use', 'true', 'false', 'null', 'nil',
                       'self', 'this', 'super', 'async', 'await', 'yield', 'match', 'case', 'switch', 'break',
                       'continue', 'pass', 'in', 'is', 'not', 'and', 'or', 'type', 'typeof', 'instanceof'])

# (extract_identifier_info and find_identifier_usages remain the same)
# These regex-based functions are still the weak point for complex code/languages.
# AST-based analysis would be the primary way to significantly improve robustness here.
def extract_identifier_info(code: str, lang: str) -> dict:
    identifier_info = {}
    # ... (existing logic) ...
    def add_identifier(name, context, expression=None):
        if name and name not in RESERVED_KEYWORDS and not name.isdigit() and name not in identifier_info:
             identifier_info[name] = {
                 "context": context, "expressions": [expression.strip()] if expression else [], "usages": []
             }
        elif name in identifier_info and expression:
             if 'expressions' not in identifier_info[name]: identifier_info[name]['expressions'] = []
             identifier_info[name]['expressions'].append(expression.strip())
    # ... (rest of the function) ...
    assignments = re.findall(IDENTIFIER_PATTERNS["variable_assignment"], code)
    for var in assignments: add_identifier(var, "variable_assignment", "unknown_expression")
    func_param_defs = re.findall(IDENTIFIER_PATTERNS["function_parameter"], code)
    for params_str in func_param_defs:
        params = params_str.split(',')
        for param in params:
            param = param.strip();
            if not param: continue
            match = re.search(r'(?:[\w\:]+\s+)?([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*[:=\[\]].*)?$', param)
            if match: add_identifier(match.group(1), "parameter")
    func_defs = re.findall(IDENTIFIER_PATTERNS["function_definition"], code)
    for func_name in func_defs: add_identifier(func_name, "function_definition")
    class_defs = re.findall(IDENTIFIER_PATTERNS["class_definition"], code)
    for class_name in class_defs: add_identifier(class_name, "class_definition")
    loop_vars = re.findall(IDENTIFIER_PATTERNS["loop_variable"], code)
    for var in loop_vars: add_identifier(var, "loop_variable")
    loop_iters = re.findall(IDENTIFIER_PATTERNS["loop_iterator"], code)
    for var in loop_iters: add_identifier(var, "loop_iterator")
    found_identifiers = list(identifier_info.keys())
    for ident in found_identifiers: identifier_info[ident]["usages"] = find_identifier_usages(code, ident)
    logger.debug(f"Extracted identifiers: {list(identifier_info.keys())}")
    return identifier_info

def find_identifier_usages(code: str, identifier: str) -> list:
    usages = []
    ident_pattern = r'\b' + re.escape(identifier) + r'\b'
    # ... (existing regex logic) ...
    if re.search(rf'{ident_pattern}\s*=[^=]', code): usages.append({"type": "assignment_lhs"})
    if re.search(rf'[^=]=\s*{ident_pattern}', code): usages.append({"type": "assignment_rhs"})
    if re.search(rf'{ident_pattern}\s*[\+\-\*\/]', code) or re.search(rf'[\+\-\*\/]\s*{ident_pattern}', code): usages.append({"type": "arithmetic"})
    if re.search(rf'{ident_pattern}\s*(?:==|!=|<=|>=|<|>|===|!==)', code) or re.search(rf'(?:==|!=|<=|>=|<|>|===|!==)\s*{ident_pattern}', code): usages.append({"type": "comparison"})
    if re.search(rf'\w+\s*\([^)]*{ident_pattern}[^)]*\)', code):
         if not re.search(rf'\b{identifier}\s*\(', code): usages.append({"type": "function_argument"})
    if re.search(rf'{ident_pattern}\s*\(', code): usages.append({"type": "function_call"})
    if re.search(rf'return\s+{ident_pattern}', code): usages.append({"type": "return"})
    if re.search(rf'(?:if|while|elif|case)\s*\([^)]*{ident_pattern}[^)]*\)', code): usages.append({"type": "conditional_expression"})
    if re.search(rf'{ident_pattern}\.\w+', code): usages.append({"type": "property_method_access_lhs"})
    if re.search(rf'\w+\.{ident_pattern}', code): usages.append({"type": "property_method_access_rhs"})
    if re.search(rf'{ident_pattern}\s*\[', code): usages.append({"type": "collection_access"})
    return usages


def calculate_naming_metrics(identifier_info: dict, original_code: str, current_map: dict, initial_map: dict) -> dict:
    """
    Calculates naming metrics. Refined penalties/bonuses and improvement score calculation.
    """
    metrics = {
        "semantics_score": 0.0, "consistency_score": 0.0, "descriptiveness_score": 0.0,
        "context_score": 0.0, "best_practices_score": 0.0, "improvement_score": 0.0
    }
    if not identifier_info: return metrics
    identifier_names = list(identifier_info.keys())
    if not identifier_names: return metrics
    num_identifiers = len(identifier_names)

    # --- Semantic & Descriptiveness Score (Combined logic slightly) ---
    semantic_descriptive_scores = []
    common_patterns = get_common_naming_patterns() # Use updated patterns
    for ident, info in identifier_info.items():
        score = 0.5
        context = info.get("context", "unknown")

        # Apply common patterns (more impact now)
        for pattern, modifier in common_patterns.items():
             if re.search(pattern, ident, re.IGNORECASE):
                  score += modifier * 1.2 # Amplify effect of good/bad patterns

        # Length penalties/bonuses
        if len(ident) < 3 and ident.lower() not in ['i', 'j', 'k', 'x', 'y', 'z', 'id', 'db', 'io', 'ui']: score -= 0.35 # Stricter short names
        elif len(ident) > 30: score -= 0.3 # Increased penalty for long names
        elif len(ident) > 20: score -= 0.15 # Penalty for moderately long

        # Context specific checks (slightly refined)
        if context == "function_definition":
            is_verb_ish = any(ident.lower().startswith(verb) for verb in ["get", "set", "calc", "proc", "handle", "is", "has", "val", "parse", "load", "save", "create", "update", "delete", "rend", "exec", "run", "build", "init"])
            is_noun_ish = any(ident.lower().endswith(n) for n in ["er", "or", "tion", "ment", "ity", "data", "list", "map", "set", "config", "util"])
            if is_verb_ish and not is_noun_ish: score += 0.15 # Reward verb-like function names
            elif is_noun_ish and not is_verb_ish: score -= 0.1 # Slightly penalize pure noun functions unless clear reason

        elif context == "class_definition":
            if not re.match(r'^[A-Z]', ident): score -= 0.25 # Stronger penalty
            if not ident[0].isupper() or not any(c.islower() for c in ident[1:]): # Penalize ALL_CAPS class names unless very short
                 if len(ident) > 4: score -= 0.1

        # Stronger penalties for generic/bad names
        if "mutated" in ident.lower(): score -= 0.5 * ident.lower().count("mutated")
        if re.match(r'^(var|temp|tmp|data|val|value|obj|o|myvar|my_var)\d*$', ident.lower()): score -= 0.4

        semantic_descriptive_scores.append(max(0, min(1.0, score)))

    avg_sem_desc_score = sum(semantic_descriptive_scores) / num_identifiers if semantic_descriptive_scores else 0
    # Assign this combined score to both, maybe adjust weights later if needed
    metrics["semantics_score"] = avg_sem_desc_score
    metrics["descriptiveness_score"] = avg_sem_desc_score
    # Context score can be derived from usage analysis (kept simple here)
    metrics["context_score"] = calculate_context_score(identifier_info) # Separate function for clarity


    # --- Consistency Score ---
    metrics["consistency_score"] = calculate_naming_consistency(identifier_names) # Use existing function

    # --- Best Practices Score ---
    bp_scores = []
    for ident in identifier_names:
        bp_score = 0.5
        if len(ident) == 1 and ident.lower() not in ['i', 'j', 'k', 'x', 'y', 'z', 'n', 'm', 'c', 'e', 'f', 'g', 'a', 'b', 'd', 'p', 'q', 'r', 's', 't', 'v', 'w']: # Allow more common single letters
             bp_score -= 0.2
        if re.match(r'^[a-zA-Z]+[0-9]+$', ident) and not re.match(r'^(utf|ascii|iso|md|sha|http)\d+$', ident.lower()): bp_score -= 0.25
        if ident.lower() in ["temp", "tmp", "var", "value", "val", "foo", "bar", "data", "obj", "o", "myvar", "detail", "info"]: bp_score -= 0.3 # Stronger penalty
        if 4 <= len(ident) <= 18: bp_score += 0.1 # Tighter optimal length
        elif len(ident) > 25: bp_score -= 0.25
        elif len(ident) < 3: bp_score -= 0.1 # Penalize length 2 unless allowed single char

        if "_" in ident and any(c.isupper() for c in ident): bp_score -= 0.1 # Penalize mixed snake_Case and CamelCase

        bp_scores.append(max(0, min(1.0, bp_score)))
    metrics["best_practices_score"] = sum(bp_scores) / num_identifiers if bp_scores else 0


    # --- Improvement Score (Refined) ---
    # Reward more significant changes (using edit distance)
    changed_count = 0
    total_distance = 0
    significant_change_count = 0
    max_possible_distance = 0 # Theoretical max change

    original_keys_in_initial = set(initial_map.keys())
    original_keys_in_current = set(current_map.keys())
    relevant_keys = original_keys_in_initial.intersection(original_keys_in_current)

    if not relevant_keys: # No common base to compare
         metrics["improvement_score"] = 0.0
         return metrics

    for orig_key in relevant_keys:
         initial_name = str(initial_map.get(orig_key, ""))
         current_name = str(current_map.get(orig_key, ""))

         if initial_name and current_name and initial_name != current_name:
             changed_count += 1
             distance = levenshtein_distance(initial_name, current_name)
             total_distance += distance
             max_possible_distance += max(len(initial_name), len(current_name)) # Approximation

             # Define 'significant' change: distance > 2 and score improved
             if distance > 2 and score_variable_name(current_name) > score_variable_name(initial_name):
                  significant_change_count += 1

    num_relevant = len(relevant_keys)
    # Combine ratio of changed, ratio of significant changes, and normalized distance
    change_ratio = changed_count / num_relevant
    significant_ratio = significant_change_count / num_relevant
    distance_ratio = (total_distance / max_possible_distance) if max_possible_distance > 0 else 0

    # Give more weight to significant changes and distance
    metrics["improvement_score"] = (0.2 * change_ratio +
                                    0.5 * significant_ratio +
                                    0.3 * distance_ratio)
    metrics["improvement_score"] = max(0.0, min(1.0, metrics["improvement_score"])) # Clamp

    return metrics

def calculate_context_score(identifier_info: dict) -> float:
    """Calculates score based on how well name matches usage context."""
    # This remains heuristic without deeper analysis (AST)
    # Placeholder - Can be expanded based on find_identifier_usages results
    context_scores = []
    if not identifier_info: return 0.0
    for ident, info in identifier_info.items():
         score = 0.5
         usages = info.get("usages", [])
         # Example: If used in arithmetic, reward numeric/operand names
         if any(u['type'] == 'arithmetic' for u in usages):
             if any(term in ident.lower() for term in ["num", "val", "operand", "term", "factor", "sum", "diff", "prod", "quot"]): score += 0.2
             else: score -= 0.1
         # Example: If used as function call, reward verb-like names
         if any(u['type'] == 'function_call' for u in usages):
             if any(ident.lower().startswith(v) for v in ["get", "set", "calc", "proc", "is", "has", "run", "build"]): score += 0.15
             else: score -= 0.05
         # Example: If collection access involved, reward plural or collection names
         if any(u['type'] == 'collection_access' for u in usages):
              if ident.lower().endswith('s') or any(c in ident.lower() for c in ["list", "map", "dict", "set", "arr", "coll", "items", "elements"]): score += 0.2
              else: score -= 0.1

         context_scores.append(max(0.0, min(1.0, score)))

    return sum(context_scores) / len(context_scores) if context_scores else 0.0


# Levenshtein distance for Improvement Score
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


# (calculate_naming_consistency remains the same)
def calculate_naming_consistency(identifier_names: list) -> float:
    if not identifier_names or len(identifier_names) < 2: return 1.0
    casing_patterns = {
        "PascalCase": r'^[A-Z][a-zA-Z0-9]*$', "camelCase": r'^[a-z][a-zA-Z0-9]*$',
        "snake_case": r'^[a-z][a-z0-9_]*[a-z0-9]$', "UPPER_SNAKE": r'^[A-Z][A-Z0-9_]*[A-Z0-9]$',
        "kebab-case": r'^[a-z][a-z0-9\-]*[a-z0-9]$',
    }
    style_counts = {style: 0 for style in casing_patterns}; unmatched_count = 0
    for ident in identifier_names:
         if len(ident) <= 1: continue
         matched = False
         for style, pattern in casing_patterns.items():
             if re.match(pattern, ident):
                 if style == "camelCase" and re.match(casing_patterns["PascalCase"], ident): continue
                 if style == "snake_case" and re.match(casing_patterns["UPPER_SNAKE"], ident): continue
                 style_counts[style] += 1; matched = True; break
         if not matched: unmatched_count += 1
    total_considered = len([ident for ident in identifier_names if len(ident)>1])
    if total_considered == 0: return 1.0
    if not style_counts or all(v == 0 for v in style_counts.values()): return 0.0
    dominant_count = max(style_counts.values())
    consistency = dominant_count / total_considered
    return consistency


def get_common_naming_patterns() -> dict:
    """Updated patterns with refined modifiers."""
    return {
        # --- Strong Positives ---
        r'^(?:is|has|can|should|allow|enable|check|validate)': 0.35, # Boolean prefixes
        r'index|idx|pos|offset': 0.25,
        r'count|total|sum|average|mean|aggregate': 0.3,
        r'min|max|limit|bound|range': 0.25,
        r'get|fetch|retrieve|load': 0.2, # Getter verbs
        r'set|update|assign|store|save': 0.2, # Setter verbs
        r'add|append|insert|push': 0.2, # Add verbs
        r'remove|delete|pop|clear': 0.2, # Remove verbs
        r'create|build|generate|make': 0.2, # Creation verbs
        r'process|handle|execute|run|perform|invoke': 0.25, # Action verbs
        r'parse|decode|extract': 0.25,
        r'convert|transform|map|format': 0.25,
        r'render|display|show|draw|paint': 0.2,
        r'config|setting|option|pref': 0.25, # Configuration
        r'context|env|session|state': 0.2, # Context/State
        r'manager|controller|service|provider|repository|factory|builder': 0.25, # Design patterns
        r'handler|listener|observer|subscriber|interceptor|callback|delegate': 0.25, # Event patterns
        r'util|helper|support|common|shared': 0.15, # Utilities (slightly less score)
        r'model|view|presenter|entity|dto|record|struct|schema': 0.25, # Data structures
        r'node|edge|vertex|graph|tree|root|parent|child|leaf': 0.25, # Graph/Tree terms
        r'list|array|vector|sequence|items|elements|collection': 0.25, # Collections
        r'map|dict|table|lookup|cache|index': 0.25, # Mappings
        r'queue|stack|deque|buffer': 0.25, # Queues/Stacks
        r'key|id|identifier|uuid|token': 0.25, # Identifiers
        r'name|label|title|desc': 0.2, # Naming/Descriptions
        r'path|url|uri|route': 0.25, # Paths/URLs
        r'error|exception|fault|warning|issue|log': 0.2, # Error handling
        # --- Mild Positives ---
        r'input|arg|param': 0.1,
        r'output|result|return': 0.15,
        r'value|val': 0.05, # Very generic
        r'first|last|prev|next|current|old|new': 0.15,
        r'start|end|begin|finish|init|term': 0.15,
        r'flag|status|active|ready|valid': 0.2, # Status flags
        r'size|width|height|length|dim': 0.2,
        r'operand|factor|term|coeff|ratio': 0.2, # Math terms (can be good)
        # --- Strong Negatives ---
        r'temp|tmp': -0.4,
        r'foo|bar|baz|qux': -0.5,
        r'data|obj|info|detail|item|element': -0.3, # Overly generic nouns
        r'^(var|val|temp|tmp|data|xxx|stuff|things)\d*$': -0.4, # Generic + number
        r'^[a-z]{1,2}$': -0.3, # Single/double lower letters (allow i,j,k etc via best prac.)
        r'^[A-Z]{1,2}$': -0.25, # Single/double upper letters
        r'^[a-z]{1,2}[0-9]+$': -0.35, # ab1, c2 etc.
        r'mutated|interim|intermediate': -0.4, # Avoid process state in name
        r'string|int|float|bool|dict|list|array': -0.2, # Avoid type names in variable names usually
        r'_+$': -0.2, # Trailing underscores (unless private convention)
        r'^_': -0.1, # Leading underscore (often convention, slight penalty unless known)
    }


#####################################
# 5. Mutation Operator (Enhanced LLM Interaction)
#####################################

def mutate_mapping_llm_enhanced(individual, lang: str, languages: dict):
    """
    Enhanced LLM mutation: Gets multiple suggestions, selects one based on novelty and quality.
    """
    current_map = individual["var_map"]
    if not current_map: return individual,
    base_code = individual["base_code"]

    try:
        current_code, _ = render_code(individual, lang, languages)
        identifier_info = extract_identifier_info(current_code, lang)
    except Exception as e:
        logger.error(f"Mutation skipped: Error rendering/extracting info: {e}")
        return individual,

    candidates = list(current_map.keys())
    if not candidates: return individual,

    num_to_mutate = random.randint(1, min(3, len(candidates)))
    keys_to_mutate = random.sample(candidates, num_to_mutate)
    logger.debug(f"Attempting to mutate original keys: {keys_to_mutate}")
    successful_mutations = 0

    for original_key in keys_to_mutate:
        current_name = str(current_map.get(original_key, ""))
        if not current_name: continue

        context_snippet = find_context_snippet(current_code, current_name)
        if not context_snippet: context_snippet = current_code # Fallback to full code

        try:
            # *** LLM Call for MULTIPLE Suggestions ***
            suggested_names = suggest_better_names( # Note plural function name
                original_name=original_key,
                current_name=current_name,
                code_context=context_snippet,
                lang=lang,
                count=3 # Ask for 3 suggestions
            )

            if not suggested_names: continue # Skip if LLM returns nothing

            # --- Selection Strategy ---
            # Score suggestions and select one. Prioritize valid, different, high-scoring names.
            best_suggestion = None
            highest_score = -1.0
            current_score = score_variable_name(current_name)

            valid_suggestions = []
            for sugg in suggested_names:
                sugg_clean = re.sub(r'^[`"\']|[`"\']$', '', str(sugg)).strip()
                # Validate identifier format and check against keywords
                if sugg_clean and sugg_clean != current_name and \
                   re.match(r'^[a-zA-Z_][a-zA-Z0-9_\-]*$', sugg_clean) and \
                   sugg_clean not in RESERVED_KEYWORDS:
                    valid_suggestions.append(sugg_clean)

            if not valid_suggestions: continue # No good suggestions

            # Score valid suggestions
            scored_suggestions = []
            for sugg in valid_suggestions:
                 sugg_score = score_variable_name(sugg)
                 # Calculate novelty (difference from current name)
                 distance = levenshtein_distance(current_name, sugg)
                 novelty_bonus = min(0.15, 0.05 * distance) # Small bonus for being different
                 combined_score = sugg_score + novelty_bonus
                 scored_suggestions.append({"name": sugg, "score": sugg_score, "combined": combined_score})

            # Sort by combined score (quality + novelty)
            scored_suggestions.sort(key=lambda x: x["combined"], reverse=True)

            # Select the best one, but maybe add randomness if scores are close?
            if scored_suggestions:
                 best_suggestion = scored_suggestions[0]["name"]
                 # Optional: If top scores are very close, pick randomly among top N?
                 # top_score = scored_suggestions[0]["combined"]
                 # close_suggestions = [s for s in scored_suggestions if s["combined"] >= top_score - 0.05]
                 # best_suggestion = random.choice(close_suggestions)["name"]

            # --- Apply Mutation ---
            if best_suggestion:
                 logger.info(f"Mutating '{original_key}': '{current_name}' -> '{best_suggestion}' (score: {scored_suggestions[0]['score']:.2f}, combined: {scored_suggestions[0]['combined']:.2f})")
                 current_map[original_key] = best_suggestion
                 successful_mutations += 1
            # else: No suitable suggestion found after filtering and scoring

        except Exception as e:
            logger.error(f"Error getting LLM suggestion list for '{current_name}': {e}")

    if successful_mutations > 0:
         del individual.fitness.values
    return individual,

# (find_context_snippet remains the same)
def find_context_snippet(code: str, identifier: str, window=5) -> str:
    lines = code.splitlines()
    ident_pattern = r'\b' + re.escape(identifier) + r'\b'
    first_occurrence_line = -1
    for i, line in enumerate(lines):
        if re.search(ident_pattern, line):
            first_occurrence_line = i; break
    if first_occurrence_line != -1:
        start = max(0, first_occurrence_line - window)
        end = min(len(lines), first_occurrence_line + window + 1)
        return "\n".join(lines[start:end])
    return None

# (score_variable_name remains largely the same - it's for quick crossover checks)
def score_variable_name(name: str) -> float:
    score = 0.5
    if not name or not isinstance(name, str): return 0.0
    name_lower = name.lower()
    # --- Penalties ---
    if "mutated" in name_lower: score -= 0.4 * name_lower.count("mutated")
    if name_lower in ["temp", "tmp", "var", "val", "foo", "bar", "baz", "data", "obj", "o", "myvar"]: score -= 0.2
    if re.match(r'^[a-zA-Z]+[0-9]+$', name) and not re.match(r'^(utf|ascii|iso|md|sha|http)\d+$', name_lower): score -= 0.15
    if len(name) == 1 and name_lower not in ['i', 'j', 'k', 'x', 'y', 'z', 'n', 'm', 'c', 'e', 'f', 'g', 'a', 'b', 'd', 'p', 'q', 'r', 's', 't', 'v', 'w']: score -= 0.2
    # --- Bonuses ---
    if 4 <= len(name) <= 18: score += 0.1 # Adjusted length bonus
    elif len(name) > 25: score -= 0.15
    else: score -= 0.05 # Penalty for 2, 3 or 19-25 length
    common_terms = [...] # Use list from get_common_naming_patterns keys if needed, or keep simpler list
    # Simplified check for crossover score:
    if any(term in name_lower for term in ["count", "total", "index", "result", "value", "list", "map", "node", "flag", "is", "has", "get", "set", "add", "process"]): score += 0.1
    if re.match(r'^[A-Z]', name) and not name.isupper(): score += 0.05 # PascalCase
    elif re.match(r'^[a-z]', name) and '_' in name: score += 0.05 # snake_case
    elif re.match(r'^[a-z]', name) and not '_' in name: score += 0.05 # camelCase
    return max(0.0, min(1.0, score))


#####################################
# 6. Improved Crossover Operator (Adjusted Threshold)
#####################################
def improved_crossover_mapping(ind1, ind2):
    """
    Intelligent crossover. Increased threshold slightly to avoid swapping for minimal score diffs.
    """
    map1 = ind1["var_map"]
    map2 = ind2["var_map"]
    new_map1 = {}
    new_map2 = {}
    all_vars = set(map1.keys()) | set(map2.keys())

    # --- Score Difference Threshold ---
    SCORE_DIFF_THRESHOLD = 0.15 # Was 0.1 - requires a more significant difference to force swap

    for var_key in all_vars:
        name1 = map1.get(var_key)
        name2 = map2.get(var_key)
        name1_str = str(name1) if name1 is not None else ""
        name2_str = str(name2) if name2 is not None else ""
        score1 = score_variable_name(name1_str) if name1_str else 0
        score2 = score_variable_name(name2_str) if name2_str else 0

        if name1_str and name2_str:
            if score1 > score2 + SCORE_DIFF_THRESHOLD:
                chosen_name = name1_str
            elif score2 > score1 + SCORE_DIFF_THRESHOLD:
                chosen_name = name2_str
            else: # Scores are close or equal
                # Keep original names more often when scores are close (50/50 chance)
                if random.random() < 0.5:
                    new_map1[var_key] = name1_str
                    new_map2[var_key] = name2_str
                else: # Or choose the slightly better one / randomly
                    chosen_name = name1_str if score1 >= score2 else name2_str
                    new_map1[var_key] = chosen_name
                    new_map2[var_key] = chosen_name
                continue # Skip common assignment below

            new_map1[var_key] = chosen_name
            new_map2[var_key] = chosen_name

        elif name1_str:
            new_map1[var_key] = name1_str
            new_map2[var_key] = name1_str
        elif name2_str:
            new_map1[var_key] = name2_str
            new_map2[var_key] = name2_str

    ind1["var_map"] = new_map1
    ind2["var_map"] = new_map2
    del ind1.fitness.values
    del ind2.fitness.values
    return ind1, ind2


#####################################
# 7. GA Setup and Execution (Adjusted Parameters)
#####################################
def run_stage1_ga_rename(initial_code: str, lang: str, languages: dict, population_size: int = 20, generations: int = 15, initial_map_retries=2):
    """
    Runs the GA with adjusted parameters for continued improvement.
    - Higher base mutation rate.
    - More aggressive adaptive mutation increase.
    - Uses enhanced LLM mutation.
    """
    initial_map = None
    # ... (initial map loading logic remains same) ...
    for attempt in range(initial_map_retries):
        try:
            logger.info(f"Attempting to get initial mapping (attempt {attempt+1}/{initial_map_retries})...")
            initial_map = get_variable_mapping(initial_code, lang)
            if initial_map:
                 initial_map = {str(k): str(v) for k, v in initial_map.items() if v}
                 break
        except Exception as e:
            logger.error(f"Could not obtain initial mapping (attempt {attempt+1}): {e}")
            if attempt == initial_map_retries - 1:
                 logger.error("Failed to get initial mapping. Aborting.")
                 return initial_code
    if not initial_map:
         logger.warning("Proceeding without an initial LLM mapping.")
         initial_map = {}

    logger.info(f"Initial Mapping ({len(initial_map)} identifiers): {initial_map}")
    pop = init_population(population_size, initial_code, initial_map)
    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate_individual, lang=lang, languages=languages, initial_map=initial_map)
    toolbox.register("mate", improved_crossover_mapping)
    # Use the *new* enhanced mutation operator
    toolbox.register("mutate", mutate_mapping_llm_enhanced, lang=lang, languages=languages)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Evaluate initial population
    logger.info("Evaluating initial population...")
    try:
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit
    except Exception as e:
        logger.error(f"Error evaluating initial population: {e}. Cannot proceed.")
        return initial_code

    # --- Adjusted GA Parameters ---
    base_mutation_prob = 0.35 # Higher base mutation chance
    mutation_prob = base_mutation_prob
    stagnation_counter = 0
    max_stagnation = 3 # Trigger increase faster
    mutation_increase_factor = 0.10 # Increase mutation rate more sharply
    max_mutation_prob = 0.85 # Allow mutation rate to go higher

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda f: sum(v[0] for v in f) / len(f))
    stats.register("std", lambda f: (sum((v[0] - (sum(x[0] for x in f) / len(f)))**2 for v in f) / len(f))**0.5)
    stats.register("min", lambda f: min(v[0] for v in f))
    stats.register("max", lambda f: max(v[0] for v in f))

    logger.info("Starting GA evolution...")
    last_best_overall_score = -1.0 # Initialize differently

    for gen in range(generations):
        # --- Standard GA Operations ---
        offspring = toolbox.select(pop, len(pop))
        offspring = [creator.Individual(ind) for ind in offspring]

        # Crossover (adjust probability slightly?)
        crossover_prob = 0.6
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < crossover_prob:
                toolbox.mate(offspring[i], offspring[i+1])

        # Mutation (Adaptive)
        mutated_count = 0
        for i in range(len(offspring)):
             if random.random() < mutation_prob:
                  toolbox.mutate(offspring[i])
                  mutated_count += 1

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
             fitnesses = list(map(toolbox.evaluate, invalid_ind))
             for ind, fit in zip(invalid_ind, fitnesses):
                 ind.fitness.values = fit

        # Update Hall of Fame *before* checking stagnation
        hof.update(offspring)
        current_best_overall_score = hof[0].fitness.values[0] # Best ever found

        # Replace population
        pop[:] = offspring

        # --- Statistics and Logging ---
        record = stats.compile(pop)
        best_current_gen_score = record['max']
        logger.info(f"Gen {gen+1}/{generations}: "
                    f"Best Score: {best_current_gen_score:.4f} (Overall: {current_best_overall_score:.4f}), "
                    f"Avg: {record['avg']:.4f}, Min: {record['min']:.4f}, "
                    f"Mutated: {mutated_count}, MutProb: {mutation_prob:.3f}") # More precision

        # --- Adaptive Mutation Rate Adjustment (More Aggressive) ---
        # Check if the best *overall* score has improved since the *last generation*
        improvement_threshold = 0.0001 # Require at least a tiny improvement to reset stagnation
        if gen > 0 and current_best_overall_score <= last_best_overall_score + improvement_threshold :
             stagnation_counter += 1
             logger.debug(f"Stagnation detected (counter: {stagnation_counter}). Best score: {current_best_overall_score:.4f}")
             if stagnation_counter >= max_stagnation: # Trigger increase faster
                 mutation_prob = min(max_mutation_prob, mutation_prob + mutation_increase_factor)
                 logger.debug(f"Increasing mutation probability to {mutation_prob:.3f}")
                 stagnation_counter = 0 # Reset counter after increase to allow some time
             elif stagnation_counter == 1: # Less aggressive initial increase
                  mutation_prob = min(max_mutation_prob, mutation_prob + mutation_increase_factor / 2)


        else:
             # Improvement occurred
             if gen > 0: logger.debug("Improvement detected.")
             stagnation_counter = 0
             mutation_prob = base_mutation_prob # Reset to base

        last_best_overall_score = current_best_overall_score # Store score for next gen comparison


    # --- GA finished ---
    best_ind = hof[0]
    logger.info(f"GA finished. Best overall score: {best_ind.fitness.values[0]:.4f}")
    logger.info(f"Best mapping found: {best_ind['var_map']}")
    try:
        final_code, _ = render_code(best_ind, lang, languages)
        logger.info("Final code rendered successfully.")
        return final_code
    except Exception as e:
         logger.error(f"Failed to render final code from best individual: {e}")
         return initial_code


#####################################
# 8. Main / Example Usage
#####################################
# (No changes needed in the __main__ block)
if __name__ == "__main__":
    import sys
    # Add parent dir if needed: sys.path.append('..')
    try:
        from app.parser_module import load_languages
    except ImportError:
         print("Error: Could not import app.parser_module. Make sure PYTHONPATH is set correctly or run as a module.", file=sys.stderr)
         sys.exit(1)


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
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s') # Added logger name

    try:
        langs = load_languages()
        if "python" not in langs:
            print("Error: Python language grammar not loaded.", file=sys.stderr)
            sys.exit(1)

        print("Running GA for Python code (Enhanced)...")
        best_code = run_stage1_ga_rename(
            initial_code=sample_code_py,
            lang="python",
            languages=langs,
            population_size=15, # Can adjust
            generations=10     # Can adjust
        )
        print("\n--- Initial Obfuscated Code ---")
        print(sample_code_py)
        print("\n--- Final Deobfuscated Code ---")
        if best_code: print(best_code)
        else: print("GA execution failed to produce final code.")

    except ImportError as e:
         print(f"Import Error: {e}. Make sure modules are correctly structured.", file=sys.stderr)
    except Exception as e:
         print(f"An unexpected error occurred: {e}", file=sys.stderr)
         import traceback
         traceback.print_exc()