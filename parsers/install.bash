#!/bin/bash

# Change to the parsers directory
cd "$(dirname "$0")"

# Clone all required tree-sitter repositories
git clone https://github.com/tree-sitter/tree-sitter-python.git
git clone https://github.com/tree-sitter/tree-sitter-javascript.git
git clone https://github.com/tree-sitter/tree-sitter-c.git
git clone https://github.com/tree-sitter/tree-sitter-rust.git
git clone https://github.com/tree-sitter/tree-sitter-php.git
git clone https://github.com/tree-sitter/tree-sitter-scala.git
git clone https://github.com/tree-sitter/tree-sitter-jsdoc.git
git clone https://github.com/tree-sitter/tree-sitter-css.git
git clone https://github.com/tree-sitter/tree-sitter-ql.git
git clone https://github.com/tree-sitter/tree-sitter-regex.git
git clone https://github.com/tree-sitter/tree-sitter-html.git
git clone https://github.com/tree-sitter/tree-sitter-java.git
git clone https://github.com/tree-sitter/tree-sitter-bash.git
git clone https://github.com/tree-sitter/tree-sitter-typescript.git
git clone https://github.com/tree-sitter/tree-sitter-julia.git
git clone https://github.com/tree-sitter/tree-sitter-cpp.git
git clone https://github.com/tree-sitter/tree-sitter-haskell.git
git clone https://github.com/tree-sitter/tree-sitter-c-sharp.git
git clone https://github.com/tree-sitter/tree-sitter-embedded-template.git
git clone https://github.com/tree-sitter/tree-sitter-agda.git
git clone https://github.com/tree-sitter/tree-sitter-verilog.git
git clone https://github.com/tree-sitter/tree-sitter-toml.git
git clone https://github.com/tree-sitter/tree-sitter-swift.git
git clone https://github.com/tree-sitter/tree-sitter-razor.git

echo "All tree-sitter repositories have been cloned successfully!"

#Language Supported by Self-Healing Code: AI That Reverses Code Obfuscation via Genetic Algorithms (Multi-Language Support)
# python
# javaScript
# c
# rust
# php
# scala
# jsdoc
# css
# ql
# regex
# html
# java
# bash
# typescript
# julia
# cpp
# haskell
# c#
# embedded template languages like ERB, EJS
# agda
# systemverilog
# toml
# swift
# c# razor
