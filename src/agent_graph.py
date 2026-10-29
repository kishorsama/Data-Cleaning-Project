# src/agent_graph.py
import pandas as pd
from typing import TypedDict, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

# Import your existing modular functions directly from your project files
from deduplication import remove_duplicates
from null_handling import handle_nulls
from outlier_detection import handle_outliers
# (If your functions have specific names like zscore, isolation_forest, label_encoding, minmax, import them here too)

# 1. Define the Shared State (The Central Clipboard)
class CleaningState(TypedDict):
    df: Any                          # Working Pandas DataFrame
    profile: Dict[str, Any]          # Statistical profile from profiler.py
    plan: List[Dict[str, Any]]       # Ordered cleaning steps chosen by the LLM
    user_approved: bool              # Human-in-the-Loop approval flag
    logs: List[str]                  # Audit trail of executed operations
    attempts: int                    # Loop counter to prevent infinite retries

# Structured output schema ensuring the LLM picks valid cleaning operations
class CleaningStep(BaseModel):
    operation: Literal[
        "deduplicate", "handle_nulls", "outliers_iqr", 
        "outliers_zscore", "outliers_iforest", "label_encode", "scale_minmax"
    ]
    columns: List[str] = Field(default=[], description="Target columns for this operation")
    param: float = Field(default=0.0, description="Threshold or parameter if applicable")
    reason: str = Field(description="Statistical justification for this step")

class CleaningPlan(BaseModel):
    steps: List[CleaningStep]

# 2. Define Node 1: Profiler
def profiler_node(state: CleaningState) -> Dict[str, Any]:
    df = state["df"]
    profile = {
        "total_rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {}
    }
    for col in df.columns:
        is_num = pd.api.types.is_numeric_dtype(df[col])
        profile["columns"][col] = {
            "dtype": str(df[col].dtype),
            "null_pct": round(float(df[col].isnull().mean() * 100), 2),
            "unique": int(df[col].nunique()),
            "skew": round(float(df[col].skew()), 2) if is_num and df[col].dropna().shape[0] > 0 else None
        }
    return {"profile": profile, "logs": ["Dataset profiled successfully."]}

# 3. Define Node 2: AI Planner (The LLM Brain)
def planner_node(state: CleaningState) -> Dict[str, Any]:
    # Using GPT-4o-mini (fast and cost-effective)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(CleaningPlan)
    
    prompt = f"""
    You are an Autonomous Data Preprocessing Agent.
    Analyze this dataset profile and create an ordered cleaning plan:
    {state['profile']}
    
    Rules:
    - Always deduplicate and handle nulls before outlier detection or scaling.
    - Choose 'outliers_iqr' for skewed numeric data (|skew| > 1).
    - Choose 'label_encode' for categorical object columns.
    """
    result: CleaningPlan = structured_llm.invoke(prompt)
    plan_dicts = [step.model_dump() for step in result.steps]
    return {"plan": plan_dicts, "attempts": state.get("attempts", 0) + 1}

# 4. Define Node 3: Human Review (Interrupt & Pause)
def human_review_node(state: CleaningState) -> Command[Literal["executor", "planner", "__end__"]]:
    decision = interrupt({
        "question": "Do you approve this AI cleaning plan?",
        "proposed_plan": state["plan"]
    })
    
    if decision.get("action") == "approve":
        updated_plan = decision.get("plan", state["plan"])
        return Command(update={"plan": updated_plan, "user_approved": True}, goto="executor")
    elif decision.get("action") == "replan":
        return Command(update={"user_approved": False}, goto="planner")
    else:
        return Command(goto=END)

# 5. Define Node 4: Executor (Runs your Python functions)
def executor_node(state: CleaningState) -> Dict[str, Any]:
    df = state["df"].copy()
    logs = list(state.get("logs", []))
    
    for step in state["plan"]:
        op = step["operation"]
        
        if op == "deduplicate":
            df = remove_duplicates(df)
        elif op == "handle_nulls":
            df = handle_nulls(df)
        elif op == "outliers_iqr":
            df = handle_outliers(df)
        # Add your other custom function calls here as needed
            
        logs.append(f"Successfully executed: {op} ({step['reason']})")
        
    return {"df": df, "logs": logs}

# 6. Define Node 5: Validator
def validator_node(state: CleaningState) -> Dict[str, Any]:
    df = state["df"]
    logs = list(state["logs"])
    logs.append(f"Validation complete. Final dataset shape: {df.shape}")
    return {"logs": logs}

def route_after_validator(state: CleaningState) -> Literal["planner", "__end__"]:
    if state["df"].isnull().sum().sum() > 0 and state["attempts"] < 2:
        return "planner"
    return END

# 7. Assemble the Graph
builder = StateGraph(CleaningState)

builder.add_node("profiler", profiler_node)
builder.add_node("planner", planner_node)
builder.add_node("human_review", human_review_node)
builder.add_node("executor", executor_node)
builder.add_node("validator", validator_node)

builder.add_edge(START, "profiler")
builder.add_edge("profiler", "planner")
builder.add_edge("planner", "human_review")
builder.add_edge("executor", "validator")
builder.add_conditional_edges("validator", route_after_validator)

# Compile with memory checkpointing to support the interactive pause
memory = InMemorySaver()
cleaning_graph = builder.compile(checkpointer=memory)