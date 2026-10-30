import os
import sys
import time
import uuid
from typing import TypedDict, List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field
import pandas as pd
from dotenv import load_dotenv

# Ensure local src directory is on sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

from profiler import get_dataset_profile, compare_profiles
from deduplication import deduplicate
from null_handling import handle_nulls
from outlier_detection import handle_outliers
from typecasting import typecasting
from label_encoding import label_encoding
from normalization import minmax, zscore
from compression import lossless

# Cache dataframes in memory by dataset_id to avoid serializing large DataFrames in graph checkpoints
_DATA_CACHE: Dict[str, pd.DataFrame] = {}

def get_cached_df(dataset_id: str) -> Optional[pd.DataFrame]:
    return _DATA_CACHE.get(dataset_id)

def set_cached_df(dataset_id: str, df: pd.DataFrame) -> None:
    _DATA_CACHE[dataset_id] = df

# Backward-compatible aliases
store_df = set_cached_df
retrieve_df = get_cached_df


class CleaningState(TypedDict):
    dataset_id: str
    profile_before: Dict[str, Any]
    profile_after: Optional[Dict[str, Any]]
    plan: List[Dict[str, Any]]
    autonomous_mode: bool  # True for auto-pilot, False for co-pilot review
    user_approved: bool
    logs: List[str]
    quality_report: Optional[Dict[str, Any]]
    attempts: int
    errors: List[str]


class CleaningStep(BaseModel):
    operation: Literal[
        "deduplicate", 
        "drop_uninformative",
        "typecast",
        "handle_nulls", 
        "outliers_iqr", 
        "label_encode", 
        "scale_minmax", 
        "scale_standard",
        "compress_memory"
    ]
    columns: List[str] = Field(default=[], description="Target columns. Leave empty to target all eligible columns.")
    strategy: str = Field(default="", description="Strategy argument (e.g. 'median', 'mean', 'mode', 'drop').")
    threshold: float = Field(default=1.5, description="IQR multiplier or numeric cutoff.")
    reason: str = Field(description="Brief explanation of why this operation is needed.")


class CleaningPlan(BaseModel):
    summary: str = Field(description="Quick summary of the main data quality issues identified.")
    steps: List[CleaningStep]


def build_heuristic_plan(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fallback rule-based planner used if the LLM is unreachable or disabled.
    Derives sensible cleaning steps directly from the statistical profile.
    """
    plan = []
    
    # 1. Remove duplicate rows first
    if profile.get("duplicate_rows", 0) > 0:
        plan.append({
            "operation": "deduplicate",
            "columns": [],
            "strategy": "",
            "threshold": 0.0,
            "reason": f"Drop {profile['duplicate_rows']} duplicate row(s) to remove redundant data."
        })

    # 2. Drop columns that are mostly null (>75%)
    cols = profile.get("columns", {})
    mostly_null_cols = [c for c, meta in cols.items() if meta.get("null_pct", 0) > 75.0]
    if mostly_null_cols:
        plan.append({
            "operation": "drop_uninformative",
            "columns": mostly_null_cols,
            "strategy": "drop",
            "threshold": 75.0,
            "reason": f"Remove sparse column(s) with >75% missing values: {mostly_null_cols}."
        })

    # 3. Cast string dates if detected
    date_candidates = [
        c for c, meta in cols.items() 
        if meta.get("semantic_type") == "datetime" and "datetime" not in meta.get("dtype", "")
    ]
    if date_candidates:
        plan.append({
            "operation": "typecast",
            "columns": date_candidates,
            "strategy": "datetime64[ns]",
            "threshold": 0.0,
            "reason": f"Convert string date columns to proper datetime: {date_candidates}."
        })

    # 4. Impute missing values
    null_cols = [c for c, meta in cols.items() if c not in mostly_null_cols and meta.get("null_count", 0) > 0]
    if null_cols:
        plan.append({
            "operation": "handle_nulls",
            "columns": null_cols,
            "strategy": "auto",
            "threshold": 0.0,
            "reason": "Impute missing values (median for skewed columns, mean for normal, mode for categorical)."
        })

    # 5. Clip numeric outliers
    outlier_cols = [c for c, meta in cols.items() if meta.get("outliers_count", 0) > 0 and c not in mostly_null_cols]
    if outlier_cols:
        plan.append({
            "operation": "outliers_iqr",
            "columns": outlier_cols,
            "strategy": "clip",
            "threshold": 1.5,
            "reason": f"Winsorize extreme outliers using 1.5*IQR bounds on: {outlier_cols}."
        })

    # 6. Compress memory
    if profile.get("memory_mb", 0) > 0.05:
        plan.append({
            "operation": "compress_memory",
            "columns": [],
            "strategy": "lossless",
            "threshold": 0.0,
            "reason": "Downcast numeric and categorical columns to optimize memory usage."
        })

    return plan


# -------------------------------------------------------------
# Graph Nodes
# -------------------------------------------------------------
def profiler_node(state: CleaningState) -> Dict[str, Any]:
    df = get_cached_df(state["dataset_id"])
    if df is None:
        raise ValueError(f"Dataset {state['dataset_id']} not found in memory cache.")

    profile = get_dataset_profile(df)
    logs = list(state.get("logs", []))
    logs.append(
        f"[INFO] Profiled dataset: {profile['total_rows']:,} rows, {profile['total_columns']} cols. "
        f"Initial health score: {profile['quality_score']}/100"
    )
    return {"profile_before": profile, "logs": logs}


def planner_node(state: CleaningState) -> Dict[str, Any]:
    profile = state["profile_before"]
    logs = list(state.get("logs", []))
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    plan = []

    # Try LLM planner if API key is present
    if api_key:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,
                google_api_key=api_key,
                request_timeout=15
            )
            structured_llm = llm.with_structured_output(CleaningPlan)

            prompt = f"""
            You are an autonomous data engineering agent. Review this dataset summary and output an ordered cleaning plan.

            Dataset Profile:
            {profile}

            Plan requirements:
            - Deduplicate first if duplicate_rows > 0.
            - Drop uninformative columns if null_pct > 75%.
            - Typecast string dates to datetime if found.
            - Handle remaining missing values using 'handle_nulls' (strategy: auto, median, mean, or mode).
            - Clip outliers using 'outliers_iqr' for numeric columns with notable outlier counts.
            - Apply 'compress_memory' to reduce memory usage.
            """
            result: CleaningPlan = structured_llm.invoke(prompt)
            plan = [step.model_dump() for step in result.steps]
            logs.append(f"[PLAN] LLM proposed {len(plan)} step(s): {result.summary}")
        except Exception as e:
            logs.append(f"[WARN] LLM unavailable ({e}). Falling back to heuristic planner.")
            plan = build_heuristic_plan(profile)
    else:
        logs.append("[INFO] No API key configured. Using heuristic planner.")
        plan = build_heuristic_plan(profile)

    if not plan:
        plan = build_heuristic_plan(profile)

    return {
        "plan": plan, 
        "logs": logs, 
        "attempts": state.get("attempts", 0) + 1
    }


def human_review_node(state: CleaningState) -> Command[Literal["executor", "planner", "__end__"]]:
    # In auto-pilot mode, bypass review and execute directly
    if state.get("autonomous_mode", True):
        return Command(update={"user_approved": True}, goto="executor")

    # In co-pilot mode, interrupt and pause for user input
    review = interrupt({
        "question": "Review and approve the proposed cleaning plan.",
        "plan": state["plan"],
        "profile": state["profile_before"]
    })

    if isinstance(review, dict) and review.get("action") == "approve":
        updated_plan = review.get("plan", state["plan"])
        return Command(update={"plan": updated_plan, "user_approved": True}, goto="executor")
    elif isinstance(review, dict) and review.get("action") == "replan":
        return Command(update={"user_approved": False}, goto="planner")
    else:
        return Command(goto=END)


def executor_node(state: CleaningState) -> Dict[str, Any]:
    dataset_id = state["dataset_id"]
    df = get_cached_df(dataset_id).copy()
    logs = list(state.get("logs", []))
    errors = list(state.get("errors", []))

    for idx, step in enumerate(state["plan"], 1):
        op = step.get("operation")
        cols = step.get("columns", [])
        strategy = step.get("strategy", "")
        threshold = step.get("threshold", 1.5)
        reason = step.get("reason", "")

        t0 = time.time()
        try:
            if op == "deduplicate":
                df = deduplicate(df)
            elif op == "drop_uninformative":
                drop_targets = [c for c in cols if c in df.columns]
                if drop_targets:
                    df = df.drop(columns=drop_targets)
            elif op == "typecast":
                cast_targets = [c for c in cols if c in df.columns]
                if cast_targets:
                    df = typecasting(df, cast_targets, strategy or "datetime64[ns]")
            elif op == "handle_nulls":
                df = handle_nulls(df, strategy=strategy or "auto", columns=cols or None)
            elif op == "outliers_iqr":
                df = handle_outliers(df, columns=cols or None, threshold=threshold)
            elif op == "label_encode":
                cat_cols = [c for c in cols if c in df.columns]
                if cat_cols:
                    df, _ = label_encoding(df, cat_cols)
            elif op == "scale_minmax":
                num_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                if num_cols:
                    df = minmax(df, num_cols)
            elif op == "scale_standard":
                num_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                if num_cols:
                    df = zscore(df, num_cols)
            elif op == "compress_memory":
                df = lossless(df)

            ms = round((time.time() - t0) * 1000, 1)
            logs.append(f"[EXEC] Step {idx}: '{op}' done in {ms}ms - {reason}")
        except Exception as ex:
            err = f"[ERROR] Step {idx} '{op}' failed: {ex}"
            logs.append(err)
            errors.append(err)

    set_cached_df(dataset_id, df)
    return {"logs": logs, "errors": errors}


def validator_node(state: CleaningState) -> Dict[str, Any]:
    dataset_id = state["dataset_id"]
    df = get_cached_df(dataset_id)
    profile_before = state["profile_before"]
    profile_after = get_dataset_profile(df)
    report = compare_profiles(profile_before, profile_after)

    logs = list(state.get("logs", []))
    logs.append(
        f"[VALIDATE] Health score changed from {report['quality_score_before']} to {report['quality_score_after']} "
        f"(+{report['score_improvement']} pts). Removed {report['nulls_eliminated']} null(s), "
        f"{report['duplicates_eliminated']} duplicate(s), saved {report['memory_saved_pct']}% memory."
    )

    return {
        "profile_after": profile_after,
        "quality_report": report,
        "logs": logs
    }


def route_after_validator(state: CleaningState) -> Literal["planner", "__end__"]:
    df = get_cached_df(state["dataset_id"])
    has_nulls = int(df.isnull().sum().sum()) > 0 if df is not None else False
    attempts = state.get("attempts", 0)

    # Re-enter planner once if nulls still exist after the first pass
    if has_nulls and attempts < 2:
        return "planner"
    return END


# -------------------------------------------------------------
# Graph Wiring
# -------------------------------------------------------------
workflow = StateGraph(CleaningState)

workflow.add_node("profiler", profiler_node)
workflow.add_node("planner", planner_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("executor", executor_node)
workflow.add_node("validator", validator_node)

workflow.add_edge(START, "profiler")
workflow.add_edge("profiler", "planner")
workflow.add_edge("planner", "human_review")
workflow.add_edge("executor", "validator")
workflow.add_conditional_edges("validator", route_after_validator)

cleaning_graph = workflow.compile(checkpointer=InMemorySaver())


def run_autonomous_clean(df: pd.DataFrame, autonomous_mode: bool = True) -> Dict[str, Any]:
    """
    Programmatic helper to run the full cleaning pipeline on a DataFrame.
    """
    dataset_id = str(uuid.uuid4())
    set_cached_df(dataset_id, df.copy())
    config = {"configurable": {"thread_id": dataset_id}}

    initial_state = {
        "dataset_id": dataset_id,
        "autonomous_mode": autonomous_mode,
        "attempts": 0,
        "logs": [],
        "errors": []
    }

    final_state = cleaning_graph.invoke(initial_state, config=config)
    final_state["df"] = get_cached_df(dataset_id)
    return final_state
