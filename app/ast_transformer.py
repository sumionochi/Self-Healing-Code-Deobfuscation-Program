# self_healing_code/app/ast_transformer.py

import ast
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VariableRenamer(ast.NodeTransformer):
    """
    AST NodeTransformer that renames variables and function arguments based on a provided mapping.
    """
    def __init__(self, mapping: dict):
        """
        :param mapping: A dictionary mapping obfuscated variable names to descriptive names.
        """
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """
        Visit a Name node and replace its id if it's in the mapping.
        """
        if node.id in self.mapping:
            old_id = node.id
            node.id = self.mapping[node.id]
            logger.debug("Renamed variable: %s -> %s", old_id, node.id)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        """
        Visit an argument node and replace its name if it's in the mapping.
        """
        if node.arg in self.mapping:
            old_arg = node.arg
            node.arg = self.mapping[node.arg]
            logger.debug("Renamed argument: %s -> %s", old_arg, node.arg)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """
        Ensure that the function name itself remains unchanged (if needed) and then process its body.
        """
        # Optionally, you might want to rename functions too. For now, we leave function names intact.
        self.generic_visit(node)
        return node

def transform_code(code: str, mapping: dict) -> str:
    """
    Transforms the given Python code by renaming variables based on the provided mapping.
    
    Steps:
      1. Parse the code into an AST.
      2. Apply the VariableRenamer transformation.
      3. Unparse the modified AST back into source code.
    
    :param code: The original Python code.
    :param mapping: Dictionary mapping obfuscated variable names to descriptive names.
    :return: Transformed code as a string.
    """
    try:
        # Parse the original code into an AST
        tree = ast.parse(code)
        logger.info("AST parsing successful.")

        # Transform the AST using the mapping
        transformer = VariableRenamer(mapping)
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        logger.info("AST transformation complete.")

        # Convert the modified AST back to source code (requires Python 3.9+)
        new_code = ast.unparse(new_tree)
        logger.info("AST unparse successful.")
        return new_code

    except Exception as e:
        logger.error("Error during AST transformation: %s", e)
        raise e

# For production testing (optional)
if __name__ == "__main__":
    sample_code = "def x(a, b):\n    return a + b"
    # Example mapping: rename x -> add_numbers, a -> first_number, b -> second_number
    sample_mapping = {
        "x": "add_numbers",
        "a": "first_number",
        "b": "second_number"
    }
    transformed = transform_code(sample_code, sample_mapping)
    print("Transformed Code:\n", transformed)
