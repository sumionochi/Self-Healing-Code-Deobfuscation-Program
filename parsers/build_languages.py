import os
# Set compiler flags to use C11 and define static_assert appropriately.
os.environ["CFLAGS"] = "-std=c11 -Dstatic_assert=_Static_assert"

from tree_sitter import Language

def build_language_library():
    # Base directory where all tree-sitter repositories are located.
    base_path = os.path.join(os.getcwd(), "parsers")
    
    repos = [
        os.path.join(base_path, "tree-sitter-python"),
        os.path.join(base_path, "tree-sitter-javascript"),
        os.path.join(base_path, "tree-sitter-c"),
        os.path.join(base_path, "tree-sitter-php", "php"),
        os.path.join(base_path, "tree-sitter-scala"),
        os.path.join(base_path, "tree-sitter-jsdoc"),
        os.path.join(base_path, "tree-sitter-css"),
        os.path.join(base_path, "tree-sitter-ql"),
        os.path.join(base_path, "tree-sitter-regex"),
        os.path.join(base_path, "tree-sitter-html"),
        os.path.join(base_path, "tree-sitter-java"),
        os.path.join(base_path, "tree-sitter-bash"),
        os.path.join(base_path, "tree-sitter-typescript", "typescript"),
        os.path.join(base_path, "tree-sitter-julia"),
        os.path.join(base_path, "tree-sitter-haskell"),
        os.path.join(base_path, "tree-sitter-c-sharp"),
        os.path.join(base_path, "tree-sitter-embedded-template"),
        os.path.join(base_path, "tree-sitter-agda"),
        os.path.join(base_path, "tree-sitter-verilog"),
        os.path.join(base_path, "tree-sitter-toml"),
        os.path.join(base_path, "tree-sitter-swift"),
    ]

    # Ensure the build directory exists
    build_dir = os.path.join(os.getcwd(), "build")
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    # Build the shared library
    output_path = os.path.join(build_dir, "my-languages.so")
    Language.build_library(output_path, repos)
    print(f"Built language library at {output_path}")

if __name__ == "__main__":
    build_language_library()
