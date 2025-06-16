"""
Generate column name alternatives and value synonyms using the agno agentic framework.

This script reads CSV files from the specified Magellan‑style benchmarks, then
invokes an LLM agent (via agno) to craft technically correct, database‑friendly
column‑name alternatives together with value synonyms.

Usage:
    python generate_alternative_schema.py

Prerequisites:
    pip install pandas agno openai python‑dotenv

Environment variables expected (set in your .env or shell):
    OPENAI_API_KEY   # or equivalent provider key understood by agno
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from agno.agent import Agent  # ensure agno ≥0.2.0 is installed
from agno.models.nebius import Nebius
from agno.models.openai import OpenAIChat

from agno.utils.pprint import pprint_run_response

from pydantic import BaseModel, Field
from typing import List
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
np.random.seed(42)
BASE_PATH = Path("../magellan_ori")
# DATASET_NAMES = ["abt_buy", "amazon_google"]
DATASET_NAMES = ["abt_buy", "amazon_google", "beer", "company", "dblp_acm", "dblp_scholar", "fodors_zagat", "itunes_amazon", "walmart_amazon"]
SAMPLE_SIZE = 5  # how many example values to send to the LLM per column
LLM_MODEL = "gpt-4o-mini"  # pick any model your agno installation supports

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class SchemaScript(BaseModel):
    original_column_name: str = Field(..., description="the original name of the selected column")
    alternative_column_names: List[str] = Field(..., description="the alternative names of the selected column.")

def sample_and_serialize(df: pd.DataFrame, k: int = 5, serialize_method: str = "attribute-value-pair"):
    """Return up to *k* non-null sample rows from *df* serialized using the specified method."""
    df_nonnull = df.dropna()
    if df_nonnull.empty:
        return []

    k = min(k, len(df_nonnull))
    selected_idx = np.random.choice(df_nonnull.index, k, replace=False)
    selected_rows = df_nonnull.loc[selected_idx]

    serialized_seq = []

    for _, row in selected_rows.iterrows():
        if serialize_method == "attribute-value-pair":
            serialized = ", ".join(f"{col}: {row[col]}" for col in df.columns)
        elif serialize_method == "sentence":
            # Basic NL generation: assumes columns like name, age, gender
            sentence_parts = []
            for col in df.columns:
                val = row[col]
                if col.lower() == "name":
                    sentence_parts.insert(0, str(val))
                elif col.lower() == "age":
                    sentence_parts.append(f"is {val}-year-old")
                elif col.lower() == "gender":
                    gender_map = {"M": "male", "F": "female"}
                    sentence_parts.append(gender_map.get(str(val), str(val)))
                else:
                    sentence_parts.append(f"{col} is {val}")
            serialized = " ".join(sentence_parts) + "."
        elif serialize_method == "markdown":
            # Add header only once, at the beginning
            if not serialized_seq:
                header = "| " + " | ".join(df.columns) + " |"
                separator = "| " + " | ".join("---" for _ in df.columns) + " |"
                serialized_seq.append(header)
                serialized_seq.append(separator)
            serialized = "| " + " | ".join(str(row[col]) for col in df.columns) + " |"
        else:
            raise ValueError(f"Unsupported serialize_method: {serialize_method}")

        serialized_seq.append(serialized)

    return serialized_seq

def _build_prompt(df: pd.DataFrame, dataset_name: str, column: str, examples: str | int | float, n_alt: int) -> str:
    """Craft a concise, instruction‑oriented prompt for the agent."""
    columns_name_concat = ", ".join(df.columns.tolist())
    return (
        f"Given the column {column!r} of table named {dataset_name}, "
        f"generate {n_alt} alternative column names that adhere to typical "
        f"database naming conventions such as underscores and abbreviations. \n"
        f"The table has the following columns: {columns_name_concat}, respectively. \n"
        f"The sample rows in the table is {examples}\n"
        # f"Return the answer as JSON where each element contains:\n"
        # f"  {dataset_name}: {column!r} : list[str]\n"
    )


# ---------------------------------------------------------------------------
# Main processing routine
# ---------------------------------------------------------------------------

def generate_for_dataset(dataset_name: str, agent: Agent):
    """Process all ″table*.csv″ files under *dataset_name* and return a dict."""
    dataset_path = BASE_PATH / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    results: dict[str, dict[str, list[dict[str, object]]]] = {}

    for csv_path in dataset_path.glob("table*.csv"):
        if "train" in csv_path.name or "test" in csv_path.name:
            continue
        df = pd.read_csv(csv_path)
        n_rows = len(df)
        n_alt = max(1, int(n_rows / 100 * 2))
        n_alt = min(n_alt, 15)
        
        example_vals = sample_and_serialize(df, 3)

        for column in df.columns:
            prompt = _build_prompt(df, dataset_name, column, "\n".join(example_vals), n_alt)
            # print(prompt)
            try:
                response = agent.run(prompt)
                print(response.content.model_dump())
            except Exception as exc:
                print(f"⚠️  Agent failed on {csv_path.name}:{column}: {exc}")
                continue
            key = f"{csv_path.name}: {column}"
            results[key] = response.content.model_dump()  # expecting JSON‑parseable structure
            # break

    return results


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


from dotenv import load_dotenv
import os
load_dotenv(dotenv_path="../../.env")

def main() -> None:
    agent = Agent(
        model=Nebius(id="deepseek-ai/DeepSeek-V3-0324", api_key=os.getenv("NEBIUS_API_KEY")),
        description="You are a meticulous data engineer specialising in schema design, schema matching and data integration. ",
        instructions=["Follow the user's instructions exactly. Respond only with valid JSON."],
        response_model=SchemaScript,
        use_json_mode=True,
    )

    all_outputs = {}
    for ds in DATASET_NAMES:
        print(f"▶︎ Processing dataset: {ds}")
        results = generate_for_dataset(ds, agent)
        
        for key in results.keys():
            name = ds+"_"+key
            all_outputs[name] = results[key]

    with open("output_concise.json", "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)

    print(f"✔︎ Completed. Results written to output_concise.json")


if __name__ == "__main__":
    main()
