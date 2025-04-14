import logging
import uuid
import time
import difflib
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, Path as FastApiPath
from pydantic import BaseModel, Field
from typing import Dict, Optional
import sys

# --- Stage imports (placeholders or actual modules) ---
try:
    from app.stage1_ga_rename import run_stage1_ga_rename
    from app.stage2_llm_strings import run_stage2_llm_strings
    from app.stage3_ga_controlflow import run_stage3_ga_controlflow
    from app.stage4_llm_expressions import run_stage4_llm_expressions
    from app.stage5_llm_deadcode import run_stage5_llm_deadcode
    from app.stage6_llm_format import run_stage6_llm_format
    from app.parser_module import load_languages, parse_code
    from app.integrated_transformer import transform_code_integrated
except ImportError as e:
    print(f"Error importing stage modules: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("api_server")

app = FastAPI(
    title="Self-Healing Code Deobfuscation API (File-Based)",
    description="Matches main.py CLI: codefile, mode, interactive, output, temp_dir, population_size, generations.",
    version="1.0.0"
)

# -----------------------------------------------------------------
# Data Models replicating the CLI arguments
# -----------------------------------------------------------------
class DeobfuscationRequest(BaseModel):
    codefile: str = Field(..., description="Path to the input code file on the server machine.")
    lang: str = Field(..., description="Programming language.")
    mode: str = Field("multi_stage", description="Either 'direct' or 'multi_stage'.")
    interactive: bool = Field(False, description="True => do not auto-run multi_stage, wait for manual stage calls.")
    output: Optional[str] = Field(None, description="Optional file path to write final code.")
    temp_dir: str = Field("./deobfuscation_temp", description="Directory for intermediate outputs.")
    population_size: int = Field(15, description="Population size for GA-based stages 1 & 3.")
    generations: int = Field(10, description="Number of GA generations for stages 1 & 3.")

class JobInfo(BaseModel):
    job_id: str
    status: str
    last_successful_stage: int
    message: Optional[str] = None

class JobSubmissionResponse(JobInfo):
    final_code: Optional[str] = None  # For direct mode

class JobStatusResponse(JobInfo):
    pass

class JobResultResponse(JobInfo):
    original_file: Optional[str] = None
    final_code: Optional[str] = None

class CodeResponse(BaseModel):
    job_id: str
    stage_num: int
    code: Optional[str] = None
    message: Optional[str] = None

class DiffResponse(BaseModel):
    job_id: str
    stage_num: int
    diff: Optional[str] = None
    message: Optional[str] = None

# -----------------------------------------------------------------
# In-memory Store for multi-stage jobs
# -----------------------------------------------------------------
job_store: Dict[str, Dict] = {}

# Example:
# job_store["uuid"] = {
#   "job_id": "uuid",
#   "status": "pending_interactive",
#   "last_successful_stage": 0,
#   "message": "",
#   "params": {
#       "codefile": "/path/to/code.py",
#       "lang": "python",
#       ...
#   },
#   "stage_outputs": {
#       0: "the code from the file",
#       1: None, ...
#   }
# }

try:
    LANGUAGES = load_languages()
    logger.info("Tree-sitter languages loaded.")
except Exception as e:
    logger.error("Failed to load languages: %s", e, exc_info=True)
    LANGUAGES = None

stage_funcs = {
    1: run_stage1_ga_rename,
    2: run_stage2_llm_strings,
    3: run_stage3_ga_controlflow,
    4: run_stage4_llm_expressions,
    5: run_stage5_llm_deadcode,
    6: run_stage6_llm_format
}
stage_is_ga = {1: True, 2: False, 3: True, 4: False, 5: False, 6: False}
stage_names = {
    1: "Renaming (GA+LLM)",
    2: "String Deobfuscation (LLM)",
    3: "Control Flow Simplification (GA+LLM)",
    4: "Expression Simplification (LLM)",
    5: "Dead Code Removal (LLM)",
    6: "Formatting/Commenting (LLM)"
}

# -----------------------------------------------------------------
# Helper to read code from codefile
# -----------------------------------------------------------------
def read_codefile_or_400(codefile_path: str) -> str:
    """Reads the code from disk or raises HTTP 400 if not found/unreadable."""
    p = Path(codefile_path)
    if not p.is_file():
        raise HTTPException(status_code=400, detail=f"codefile '{codefile_path}' does not exist.")
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read codefile '{codefile_path}': {e}")

# -----------------------------------------------------------------
# Stage Execution
# -----------------------------------------------------------------
def execute_stage(job_id: str, stage_num: int, input_code: str) -> Optional[str]:
    job = job_store[job_id]
    if stage_num not in stage_funcs:
        job["status"] = "failed"
        job["message"] = f"Invalid stage number {stage_num}"
        return None

    func = stage_funcs[stage_num]
    is_ga = stage_is_ga[stage_num]

    job["status"] = f"running stage {stage_num}"
    job["message"] = None
    logger.info(f"[Job {job_id}] Starting stage {stage_num}: {stage_names[stage_num]}")

    # Build params
    params = {
        "lang": job["params"]["lang"],
        "languages": LANGUAGES
    }
    if is_ga:
        params["population_size"] = job["params"]["population_size"]
        params["generations"] = job["params"]["generations"]
        params["initial_code"] = input_code
    else:
        params["code"] = input_code

    try:
        rc = func(**params)
        if rc and isinstance(rc, str):
            job["stage_outputs"][stage_num] = rc
            job["last_successful_stage"] = stage_num
            job["status"] = "stage_complete"
            return rc
        else:
            raise RuntimeError("Stage function returned None or invalid type")
    except Exception as exc:
        logger.error(f"Stage {stage_num} failed: {exc}", exc_info=True)
        job["status"] = "failed"
        job["message"] = str(exc)
        return None

def run_full_multistage(job_id: str):
    """Runs all 6 stages in the background for a multi-stage job (non-interactive)."""
    job = job_store[job_id]
    current_code = job["stage_outputs"][0]  # the code read from the file
    for sn in range(1, 7):
        rc = execute_stage(job_id, sn, current_code)
        if rc is None:
            return
        current_code = rc
    # On success:
    job["status"] = "complete"
    job["message"] = "All stages completed successfully."

    # Possibly write final code
    outp = job["params"].get("output")
    if outp:
        try:
            Path(outp).write_text(current_code, encoding="utf-8")
        except Exception as e:
            logger.error("Failed writing final code to %s: %s", outp, e)

# -----------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------

@app.post("/deobfuscate/", response_model=JobSubmissionResponse, status_code=202)
async def deobfuscate_entrypoint(req: DeobfuscationRequest, bg: BackgroundTasks):
    """Replicates main.py CLI arguments:
       --codefile, --lang, --mode, --interactive, --output, --temp_dir, --population_size, --generations
    """
    if LANGUAGES is None:
        raise HTTPException(status_code=500, detail="Tree-sitter languages not loaded on server.")

    # 1) Read code from file, validate parse
    code_str = read_codefile_or_400(req.codefile)
    try:
        parse_code(code_str, req.lang, LANGUAGES)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Syntax check failed: {e}")

    # 2) If mode=direct => immediately do integrated transform
    if req.mode == "direct":
        logger.info("Received direct transform request.")
        try:
            result_code = transform_code_integrated(code_str, req.lang, LANGUAGES)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Direct transformation error: {e}")

        # Possibly write final code
        if req.output:
            try:
                Path(req.output).write_text(result_code, encoding="utf-8")
            except Exception as e:
                logger.error("Failed to write direct mode code to %s: %s", req.output, e)

        return JobSubmissionResponse(
            job_id="",  # no job for direct mode
            status="complete",
            last_successful_stage=0,
            message="Direct transformation complete.",
            final_code=result_code
        )

    # 3) Otherwise, multi_stage => create a job
    job_id = str(uuid.uuid4())
    logger.info(f"Creating new multi-stage job {job_id}. interactive={req.interactive}")

    job_store[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "last_successful_stage": 0,
        "message": "",
        "params": {
            "codefile": req.codefile,
            "lang": req.lang,
            "mode": req.mode,
            "interactive": req.interactive,
            "output": req.output,
            "temp_dir": req.temp_dir,
            "population_size": req.population_size,
            "generations": req.generations
        },
        "stage_outputs": {
            0: code_str,
            1: None, 2: None, 3: None,
            4: None, 5: None, 6: None
        }
    }

    if req.interactive:
        # Wait for user to call stage endpoints
        job_store[job_id]["status"] = "pending_interactive"
        job_store[job_id]["message"] = "Ready for manual stage runs."
        return JobSubmissionResponse(
            job_id=job_id,
            status="pending_interactive",
            last_successful_stage=0
        )
    else:
        # run automatically in background
        job_store[job_id]["status"] = "queued"
        job_store[job_id]["message"] = "Queued for background multi-stage execution."
        bg.add_task(run_full_multistage, job_id)

        return JobSubmissionResponse(
            job_id=job_id,
            status="queued",
            last_successful_stage=0
        )

@app.post("/deobfuscate/jobs/{job_id}/run_stage/{stage_num}", response_model=JobInfo)
async def run_stage(job_id: str, stage_num: int):
    """Run an individual stage (1-6) if interactive."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] in ("complete", "failed"):
        raise HTTPException(status_code=400, detail=f"Job status is '{job['status']}'. No further stage runs allowed.")

    required_prev = stage_num - 1
    if job["last_successful_stage"] < required_prev:
        raise HTTPException(status_code=400, detail=f"Cannot run stage {stage_num} until stage {required_prev} is done.")

    code_in = job["stage_outputs"][required_prev]
    if code_in is None:
        raise HTTPException(status_code=500, detail=f"No code from stage {required_prev} found.")

    rc = execute_stage(job_id, stage_num, code_in)
    return JobInfo(
        job_id=job_id,
        status=job_store[job_id]["status"],
        last_successful_stage=job_store[job_id]["last_successful_stage"],
        message=job_store[job_id]["message"]
    )

@app.post("/deobfuscate/jobs/{job_id}/run_remaining", response_model=JobInfo)
async def run_remaining(job_id: str, bg: BackgroundTasks):
    """Queue remaining stages for a job in the background."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] in ("complete", "failed", "running"):
        raise HTTPException(status_code=400, detail=f"Cannot run_remaining in status '{job['status']}'.")

    next_stage = job["last_successful_stage"] + 1
    if next_stage > 6:
        raise HTTPException(status_code=400, detail="No stages left to run.")

    job["status"] = "queued"
    job["message"] = f"Running stages {next_stage}-6 in background."

    def run_partial(job_id_inner: str):
        j = job_store[job_id_inner]
        code_current = j["stage_outputs"][j["last_successful_stage"]]
        for stg in range(next_stage, 7):
            rc = execute_stage(job_id_inner, stg, code_current)
            if rc is None:
                return
            code_current = rc
        j["status"] = "complete"
        j["message"] = "All stages complete."
        # Optionally write final code
        outp = j["params"].get("output")
        if outp:
            try:
                Path(outp).write_text(code_current, encoding="utf-8")
            except Exception as e:
                logger.error("Failed to write final code: %s", e)

    bg.add_task(run_partial, job_id)
    return JobInfo(**job)

@app.get("/deobfuscate/jobs/{job_id}/status", response_model=JobStatusResponse)
async def check_status(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    return JobStatusResponse(**job)

@app.get("/deobfuscate/jobs/{job_id}/result", response_model=JobResultResponse)
async def get_result(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")

    latest_stage = job["last_successful_stage"]
    final_code = job["stage_outputs"].get(latest_stage)
    return JobResultResponse(
        job_id=job_id,
        status=job["status"],
        last_successful_stage=latest_stage,
        message=job.get("message"),
        original_file=job["params"]["codefile"],
        final_code=final_code
    )

@app.get("/deobfuscate/jobs/{job_id}/code/{stage_num}", response_model=CodeResponse)
async def get_stage_code(job_id: str, stage_num: int):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    if stage_num > job["last_successful_stage"]:
        raise HTTPException(status_code=400, detail=f"Stage {stage_num} not completed yet.")
    code_out = job["stage_outputs"].get(stage_num)
    if code_out is None:
        raise HTTPException(status_code=404, detail="Stage code not found.")
    return CodeResponse(job_id=job_id, stage_num=stage_num, code=code_out)

@app.get("/deobfuscate/jobs/{job_id}/diff/{stage_num}", response_model=DiffResponse)
async def get_stage_diff(job_id: str, stage_num: int):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    if stage_num < 1 or stage_num > 6:
        raise HTTPException(status_code=400, detail="stage_num must be in [1..6].")
    if stage_num > job["last_successful_stage"]:
        raise HTTPException(status_code=400, detail=f"Stage {stage_num} not completed yet.")

    before = job["stage_outputs"].get(stage_num - 1)
    after = job["stage_outputs"].get(stage_num)
    if before is None or after is None:
        raise HTTPException(status_code=404, detail="Cannot find stage input/output for diff.")

    diff_lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"stage_{stage_num-1}",
        tofile=f"stage_{stage_num}",
        lineterm=''
    )
    diff_str = "".join(diff_lines).strip()
    if not diff_str:
        diff_str = "(No textual changes detected)"

    return DiffResponse(job_id=job_id, stage_num=stage_num, diff=diff_str)

@app.get("/")
async def root():
    return {"message": "Self-Healing Code Deobfuscation (File-based). Use /deobfuscate endpoint."}
