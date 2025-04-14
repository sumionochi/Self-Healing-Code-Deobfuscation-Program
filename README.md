# Self-Healing Code: Multi-Stage Deobfuscation Tool

This project provides **multiple ways** to deobfuscate code:

1. A **command-line** program (`main.py`) that runs the multi-stage or direct process on a single file, guided by GA + LLM and Tree-sitter parsing.
2. A **FastAPI** server (`api_server.py`) that offers an HTTP API for multi-stage or direct transformations, enabling more flexible or remote usage.
3. A **VS Code Extension** that connects to the API, provides a **graphical UI**, and handles interactive stage control, code diffs, status checks, etc., right in your editor.

<img width="1074" alt="image" src="https://github.com/user-attachments/assets/1e2f2da4-dcb2-46ae-864f-b375bf25369f" />

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage (Command-Line Interface)](#usage-command-line-interface)
  - [Examples](#examples)
  - [Multi-Stage Process](#the-multi-stage-deobfuscation-process-multi_stage-mode)
- [Supported Languages](#supported-languages)
- [Caveats and Limitations](#caveats-and-limitations)
- [API & VS Code Extension](#api--vs-code-extension)
  - [API Server Setup](#api-server-setup)
  - [API Endpoints](#api-endpoints)
  - [Using the VS Code Extension](#using-the-vs-code-extension)
  - [Workflow Example](#workflow-example)
- [FAQ](#faq)
- [Contributing](#contributing)

---

## Overview

This repository implements a **multi-stage** approach to **deobfuscating** source code in a variety of programming languages. The project combines:

- **Genetic Algorithms (GA)** for more complex transformations, guided by an LLM’s suggestions (Stages 1 & 3).
- **LLM** (e.g., GPT-4) for tasks like string decoding, expression simplification, code formatting, or removing dead code.
- **Tree-sitter** for robust syntax parsing and validation after each transformation.

**Primary Goal:** To restore readability and maintainability to intentionally obscured code.

---

## Features

- **Multi-Stage Deobfuscation**: A 6-stage pipeline that addresses distinct obfuscation techniques step by step.
- **Hybrid GA+LLM**: Applies evolutionary search to rename identifiers or simplify control flow, but uses the LLM for guidance and rendering suggestions.
- **Targeted LLM Passes**: String decoding, expression simplification, dead code removal, etc.  
- **Syntax Validation**: After each major transformation, Tree-sitter validates that the code remains syntactically correct.
- **Supports Multiple Languages**: Build the relevant Tree-sitter grammars to handle 20+ mainstream languages Python, C, Java, JavaScript, etc.
- **CLI or Automated**: You can run the tool from the command line (with interactive or automated mode). 
- **API + VS Code**: Alternatively, run a FastAPI server that offers these transformations via HTTP, and a VS Code extension that provides a GUI with live diffing, stage controls, and status updates.

---

## Technology Stack

- **Python** – Main language for the GA, LLM calls, and parser integration.
- **DEAP** – A Python library for Genetic Algorithms used in stages 1 & 3.
- **OpenAI API** – GPT-based code transformations (requires API key).
- **Tree-sitter** – For parsing code in multiple languages.  
- **FastAPI** – The web service that replicates the multi-stage logic with job tracking.  
- **VS Code Extension** – Written in TypeScript (or JavaScript) to communicate with the FastAPI server.

---

## Project Structure

```plaintext
self_healing_code/
├─ app/
│  ├─ __init__.py
│  ├─ main.py                  # CLI entry point (multi_stage or direct)
│  ├─ llm_module.py            # LLM (OpenAI) calls
│  ├─ parser_module.py         # Tree-sitter integration
│  ├─ stage1_ga_rename.py      # GA+LLM (Rename)
│  ├─ stage2_llm_strings.py    # LLM (Strings)
│  ├─ stage3_ga_controlflow.py # GA+LLM (Control Flow)
│  ├─ stage4_llm_expressions.py# LLM (Expressions)
│  ├─ stage5_llm_deadcode.py   # LLM (Dead Code)
│  ├─ stage6_llm_format.py     # LLM (Format/Comment)
│  └─ integrated_transformer.py# Single-pass transform for direct mode
├─ api_server.py               # FastAPI-based server
├─ build/
│  └─ my-languages.so          # Compiled Tree-sitter grammars
├─ input/                      # Example input code folder
├─ output/                     # Example output code folder
├─ deobfuscation_temp/         # Default folder for intermediate stage files
├─ vscode-extension/
│  ├─ package.json
│  ├─ tsconfig.json
│  ├─ src/
│  │   └─ extension.ts         # VS Code extension code
│  └─ media/
│      └─ panel.html           # UI for multi-stage control
├─ requirements.txt
├─ .env                        # Contains OPENAI_API_KEY
└─ README.md                   # This file
```

---

## Setup and Installation

1. **Clone** this repository:
   ```bash
   git clone <repo-url>
   cd self_healing_code
   ```

2. **Create a Python Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Your `requirements.txt` might include `openai`, `deap`, `python-dotenv`, `py-tree-sitter`, `fastapi`, `uvicorn`, etc.)*

4. **Set Up LLM Credentials**:
   - Create a file named `.env` in the **project root**, containing:
     ```dotenv
     OPENAI_API_KEY="your_openai_api_key_here"
     ```
   - This environment variable is used by `llm_module.py` or your GA stages to call the LLM.

5. **Build Tree-sitter Grammars**:
   - Use `py-tree-sitter` or the official `tree-sitter` CLI to compile each needed grammar into one shared library.  
   - Update `parser_module.py` so that `LIB_PATH` points to `build/my-languages.so` (or `.dll` on Windows).

---

## Usage (Command-Line Interface)

You can run **all** transformations from the command line by calling `main.py`:

```bash
python -m app.main --codefile <file> --lang <language> [options...]
```

<img width="1396" alt="Screenshot 2025-04-14 at 11 15 01 AM" src="https://github.com/user-attachments/assets/5e80be4a-9f97-498d-b8f2-30b4ffe7fb33" />

<img width="1395" alt="Screenshot 2025-04-14 at 11 18 23 AM" src="https://github.com/user-attachments/assets/7dd9a2ff-f494-4aec-8b19-a7ee5c36272a" />



**Key CLI Arguments**:

- `--codefile` (required): Path to the **obfuscated** source code file.
- `--lang` (required): The language of the code (e.g., `python`, `c`, `java`, `javascript`, etc.).
- `--mode` (optional): Either `multi_stage` (default) or `direct`.
- `--interactive` (optional, Boolean): If set, the multi-stage process will pause after each stage, letting you see diffs, confirm changes, etc.
- `--output` (optional): A file path to write the final code. If omitted, the final result prints to stdout.
- `--temp_dir` (optional): Directory for intermediate stage outputs (default: `./deobfuscation_temp`).
- `--population_size` (optional): GA population size for stages 1 & 3 (default: 15).
- `--generations` (optional): GA generations for stages 1 & 3 (default: 10).

### Examples

1. **Automated Multi-Stage (Python):**
   ```bash
   python -m app.main --codefile input/my_script.py \
       --lang python \
       --output output/my_script_clean.py
   ```
   *Reads `my_script.py`, runs all 6 stages automatically, outputs final code to `my_script_clean.py`.*

2. **Interactive Multi-Stage (C):**
   ```bash
   python -m app.main --codefile input/obf.c \
       --lang c \
       --interactive
   ```
   *Prompts you after each stage to see the diff, confirm or skip, etc.*

3. **Multi-Stage with Custom GA Settings (Java):**
   ```bash
   python -m app.main --codefile input/Source.java \
       --lang java \
       --population_size 40 \
       --generations 25 \
       --output output/Source_clean.java
   ```

4. **Direct Mode Single-Pass (JavaScript):**
   ```bash
   python -m app.main --codefile input/script.js \
       --lang javascript \
       --mode direct \
       --output output/script_clean.js
   ```
   *Uses `integrated_transformer.py` for a single attempt at deobfuscation.*

---

### The Multi-Stage Deobfuscation Process (`multi_stage` mode)

If you choose `--mode multi_stage`, `main.py` runs these 6 stages in order:

1. **Stage 1: Identifier Renaming (GA+LLM)**  
   - A Genetic Algorithm that tries renaming all identifiers to more meaningful ones, with an LLM’s suggestions guiding final picks.

2. **Stage 2: String Deobfuscation (LLM)**  
   - The LLM tries to detect and decode scrambled strings, base64 lumps, char arrays, etc.

3. **Stage 3: Control Flow Simplification (GA+LLM)**  
   - Another GA pass that attempts to simplify overly complex or contrived control flow constructs. **Experimental.**

4. **Stage 4: Expression Simplification (LLM)**  
   - LLM pass to do constant folding, boolean logic reductions, etc.

5. **Stage 5: Dead Code Removal (LLM)**  
   - LLM attempts to identify and remove obviously unreachable or unused code.

6. **Stage 6: Formatting & Commenting (LLM)**  
   - Finally, the LLM re-formats the code and may add docstrings or comments to help with readability.

If **`--interactive`** is set, you can see a textual diff after each stage and decide whether to proceed. If **`--interactive`** is not set, all 6 run automatically.

---

## Supported Languages

You can support a variety of languages by building the relevant Tree-sitter grammars. Currently, `parser_module.py` references keys for:

- `python`, `javascript`, `c`, `php`, `scala`, `jsdoc`, `css`,  
  `ql`, `regex`, `html`, `java`, `bash`, `typescript`,  
  `julia`, `haskell`, `c_sharp`, `embedded_template`,  
  `agda`, `verilog`, `toml`, `swift`, etc.

Make sure you compile the grammar for each language you wish to handle.

---

## Caveats and Limitations

- **LLM Dependence**: Results can vary widely depending on the LLM (GPT-4 vs. GPT-3.5, etc.).  
- **Semantic Equivalence**: We only check syntactic correctness. For example, Stage 3 (control flow) or Stage 5 (dead code removal) might accidentally alter program semantics. Test your code after transformation.
- **Token Costs**: Each stage calls the OpenAI API repeatedly, especially the GA-based ones. This can be expensive if your code is large or you have many generations.
- **Performance**: The GA can be CPU-heavy. Large code or high population/generations can significantly increase runtime.
- **Obfuscation Complexity**: This tool might not handle extremely specialized or advanced obfuscations.

---

## API & VS Code Extension

### API Server Setup

In addition to `main.py`, you can run a **FastAPI server** that exposes endpoints for direct or multi-stage transformations. This allows you to integrate with other tools or use the **VS Code extension**.

<img width="1384" alt="image" src="https://github.com/user-attachments/assets/866abd1d-488b-46da-998a-ee7813ef58ef" />

1. **Install** any additional dependencies in Python:
   ```bash
   pip install fastapi uvicorn
   ```
2. **Start** the server:
   ```bash
   uvicorn api_server:app --reload
   ```
   By default, it listens at `http://127.0.0.1:8000`.

3. **Check** `api_server.py` for how it reads the same codefile, language, mode, etc.  
   - You can pass JSON with `{"codefile": "...", "lang": "...", "mode": "multi_stage", ...}`.

### API Endpoints

| Method | Endpoint                                             | Description                                                          |
|:------:|:-----------------------------------------------------|:---------------------------------------------------------------------|
| **POST** | `/deobfuscate/`                                      | Accepts JSON body with `codefile`, `lang`, `mode` (`direct`/`multi_stage`), etc. |
| **GET**  | `/deobfuscate/jobs/{job_id}/status`                 | Checks the current status of a multi-stage job.                      |
| **POST** | `/deobfuscate/jobs/{job_id}/run_stage/{stage_num}`  | Runs one specific stage (1–6) for an interactive job.                |
| **GET**  | `/deobfuscate/jobs/{job_id}/code/{stage_num}`       | Retrieves code from that stage’s output.                             |
| **GET**  | `/deobfuscate/jobs/{job_id}/diff/{stage_num}`       | Shows a textual diff between stage `stage_num-1` and `stage_num`.    |
| **POST** | `/deobfuscate/jobs/{job_id}/run_remaining`          | Queues all remaining stages in the background.                       |
| **GET**  | `/deobfuscate/jobs/{job_id}/result`                 | Retrieves the final code after completion.                           |

### Using the VS Code Extension

<img width="1077" alt="image" src="https://github.com/user-attachments/assets/9e188bf4-538c-436d-84f4-1e8194548a9e" />

1. **Open** the `vscode-extension/` folder in Visual Studio Code.
2. **Install** extension dependencies:
   ```bash
   npm install
   npm install node-fetch@2
   ```
3. **Compile**:
   ```bash
   npm run compile
   ```
4. **Press F5** to launch a new “Extension Development Host” session.
5. Open the command palette and choose **“Self-Healing Code: Open Panel”** (or whatever name you set).
6. Fill out the form:
   - `codefile`: A path accessible to the server.  
   - `lang`: The language (e.g., `python`, `c`), matching keys in `parser_module.py`.  
   - `mode`: `direct` or `multi_stage`.  
   - `interactive`: True or false (for `multi_stage`).  
   - `population_size`, `generations`, etc.  
   - `output` (optional): A file path on the server to write final code.
7. Click **Submit**:
   - If **`mode=direct`**, the final code is displayed immediately.  
   - If **`mode=multi_stage`** and `interactive=false`, the job is queued. You can check status or get results.  
   - If **`mode=multi_stage`** and `interactive=true`, you can run each stage manually, view diffs, etc., from the panel.

**Retaining Webview State**:  
Your extension might set:
```ts
retainContextWhenHidden: true
```
in `createWebviewPanel` options, so the panel doesn’t reset if you switch away.

---

### Workflow Example

#### Direct Mode (API + Extension)
1. Ensure your server (`api_server.py`) is running via `uvicorn`.
2. In the extension panel, set:
   - `codefile`: `/path/to/script.js`
   - `lang`: `javascript`
   - `mode`: `direct`
   - `interactive`: (ignored in direct)
   - `population_size`, `generations`: (irrelevant in direct)
   - `output`: e.g. `/tmp/script_clean.js`
3. Click **Submit**.  
   - The extension calls `/deobfuscate/` with `{"mode": "direct" ...}`, the server returns final_code immediately, and the extension shows it.

#### Multi-Stage (Interactive)
1. **mode**: `multi_stage`, **interactive**: `true`.
2. The server creates a job but doesn’t run stages automatically.
3. You can click “Run Stage” for stage 1, “View Diff,” etc. 
4. Once you’re done with stage-by-stage, you can do “Run Remaining” or keep stepping manually until you see the final code in “result.”

---

## FAQ

1. **Will code remain semantically identical?**  
   - We only guarantee syntactic correctness. Stages 3 and 5 especially can alter logic. Always test.

2. **Why is the multi-stage mode slow?**  
   - GA-based stages do repeated calls to the LLM. Adjust population/generations or code chunking if you want faster results.

3. **Can I use file-based code in direct mode?**  
   - Yes. The server (or CLI) reads from a local file. If you want to pass code as a string, you can adapt the code for that.

4. **Does it handle extremely advanced or layered obfuscation?**  
   - Not always. Each stage tries to address certain patterns or known obfuscation techniques. In some cases, partial improvements or manual review is still required.

5. **Is there a cost to usage?**  
   - Yes, calls to OpenAI’s GPT models (used gpt-4o-mini) can be billed by tokens used. Check your OpenAI plan.

---

## Contributing

1. **Fork** the repo and create a feature branch for your changes.
2. **Open a Pull Request** with a clear description of your improvement or bugfix.
3. Ensure your changes pass existing tests (or create new tests).

Feel free to suggest new languages, better stage heuristics, or improved prompts for the LLM! 

---

Thank you for exploring the **Self-Healing Code** project! We hope it makes your deobfuscation workflows easier—whether you prefer a direct CLI approach or the convenience of an API plus a VS Code UI. Happy coding!
