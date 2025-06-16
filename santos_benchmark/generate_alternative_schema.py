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
from tqdm import tqdm
from agno.utils.pprint import pprint_run_response
from agno.models.google import Gemini

from pydantic import BaseModel, Field
from typing import List
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class SchemaScript(BaseModel):
    alternative_column_names: List[str] = Field(..., description="the alternative names of the columns.")

class EntityScript(BaseModel):
    alternative_cell_values: List[str] = Field(..., description="the alternative cell values of the selected row. each cell is a separate instance")

def sample_and_serialize(df: pd.DataFrame, table_name: str, k: int = 5, serialize_method: str = "attribute-value-pair"):
    """Return up to *k* non-null sample rows from *df* serialized using the specified method."""
    df_nonnull = df.dropna()
    if df_nonnull.empty:
        return []

    k = min(k, len(df_nonnull))
    selected_idx = np.random.choice(df_nonnull.index, k, replace=False)
    selected_rows = df_nonnull.loc[selected_idx]



    # serialized_seq = [f"This is a table named {table_name}. It contain {k} rows and {len(df.columns)} columns. The column names are {', '.join(df.columns)}, respectively. The sample rows in the table is:"]
    serialized_seq = [f"This is a table named {table_name}. It contain {k} rows and {len(df.columns)} columns. The sample rows in the table is:"]

    row_num = 0
    for n, row in selected_rows.iterrows():
        serialized_seq.append(f"The data in row {row_num} is:")
        row_num += 1
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

    return serialized_seq, selected_idx

def _build_prompt_columnnames(table_name: str, dataset_name: str, examples: str | int | float, column) -> str:
    """Craft a concise, instruction‑oriented prompt for the agent."""
    
    return (
        f"Given the column names of table named {table_name} in database named {dataset_name}, "
        f"generate alternative column names that are semantical the same and adhere to typical database naming conventions such as abbreviations. \n"
        f"The column names are: {column}, respectively. \n"
        f"The following are sample data in the table: {examples}. \n"
        f"Please keep the original column order. "
    )

def _build_prompt_entity(table_name: str, dataset_name: str, examples: str | int | float, row_value: str) -> str:
    """Craft a concise, instruction‑oriented prompt for the agent."""
    
    return (
        f"Given the a row of the table {table_name} in database named {dataset_name}, "
        f"generate alternative cell values for the row that are semantically consistent but lexically varied (remain to the same entity but present in a different way). \n"
        f"The selected row values are => {row_value}. \n"
        f"The following are sample data in the table: {examples}. \n"
        f"Please keep the original column order. "
    )


# ---------------------------------------------------------------------------
# Main processing routine
# ---------------------------------------------------------------------------

def generate_for_dataset(csv_files, agent_schema: Agent, agent_entity: Agent):
   

    results: dict[str, dict[str, list[dict[str, object]]]] = {}
    
    selected_table_num = 0
    for table_name in csv_files:
        if selected_table_num >= 3:
            break
        print(f"Processing {table_name}")
        df = pd.read_csv(f"./datalake/{table_name}")
        
        dataset_name = "_".join(table_name.split('_')[1:-1])

        sample_results = sample_and_serialize(df, table_name, 3)
        if len(sample_results) == 0:
            print(f"⚠️  {table_name} has less effect samples")
            continue
        example_vals, selected_idx = sample_results

        # build the prompt for column rename.
        column_names = ", ".join(df.columns) if len(df.columns) <= 3 else df.columns.tolist()
        prompt = _build_prompt_columnnames(table_name, dataset_name, "\n".join(example_vals), column_names)
        # print(prompt)
        # print("--------------------------------")

        if len(selected_idx) >= 3:

            example_value_select = "\n".join(["The is the first row: ", example_vals[4], "The is the second row: ", example_vals[6]])
            prompt_1 = _build_prompt_entity(table_name, dataset_name, example_value_select, example_vals[2])

            # print(prompt_1)
            # print("--------------------------------")

            example_value_select = "\n".join(["The is the first row: ", example_vals[2], "The is the second row: ", example_vals[6]])
            prompt_2 = _build_prompt_entity(table_name, dataset_name, example_value_select, example_vals[4])
            
            example_value_select = "\n".join(["The is the first row: ", example_vals[2], "The is the second row: ", example_vals[4]])
            prompt_3 = _build_prompt_entity(table_name, dataset_name, example_value_select, example_vals[6])
            
            try:
                response = agent_schema.run(prompt)
                # print(response.content.model_dump())

                response_1 = agent_entity.run(prompt_1)
                # print(response_1.content.model_dump())

                response_2 = agent_entity.run(prompt_2)

                response_3 = agent_entity.run(prompt_3)

            except Exception as exc:
                print(f"⚠️  Agent failed on {table_name}: {exc}")
                continue
            
            table_results = {"column_name": response.content.model_dump()['alternative_column_names'], "entity": [response_1.content.model_dump()["alternative_cell_values"], response_2.content.model_dump()["alternative_cell_values"], response_3.content.model_dump()["alternative_cell_values"]], "original_selected_index": selected_idx.tolist()}

            name = f"{table_name}"

            results[name] = table_results

        else:
            print(f"⚠️  {dataset_name}-{table_name} has less effect samples")
            continue

        selected_table_num += 1
        # break

    return results


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


from dotenv import load_dotenv
import os
load_dotenv(dotenv_path="../../.env")

def main() -> None:

    mapping = {}
    folder_datalake = './datalake'
    csv_files = [f for f in os.listdir(folder_datalake) if f.endswith('.csv')]

    for file in csv_files:
        key = "_".join(file.split('_')[1:-1])
        if key in mapping.keys():
            mapping[key].append(file)
        else:
            mapping[key] = [file]
    

    agent_schema = Agent(
        # model=Nebius(id="deepseek-ai/DeepSeek-V3-0324", api_key=os.getenv("NEBIUS_API_KEY")),
        model = Gemini(id="gemini-2.5-flash-preview-05-20", api_key=os.getenv("GOOGLE_API_KEY")),
        description="You are a meticulous data engineer specialising in schema design, schema matching, entity matching and data integration. ",
        instructions=["Follow the user's instructions exactly. Respond only with valid JSON."],
        response_model=SchemaScript,
        use_json_mode=True,
    )

    agent_entity = Agent(
        # model=Nebius(id="deepseek-ai/DeepSeek-V3-0324", api_key=os.getenv("NEBIUS_API_KEY")),
        model = Gemini(id="gemini-2.5-flash-preview-05-20", api_key=os.getenv("GOOGLE_API_KEY")),
        description="You are a meticulous data engineer specialising in schema design, schema matching, entity matching and data integration. ",
        instructions=["Follow the user's instructions exactly. Respond only with valid JSON."],
        response_model=EntityScript,
        use_json_mode=True,
    )


    with open("finished_names.json", "r") as f:
        finished_names = json.load(f)

    all_outputs = {}
    for ds in tqdm(mapping.keys()):
        if ds in finished_names:
            continue
        print(f"▶︎ Processing dataset: {ds}")
        results = generate_for_dataset(mapping[ds], agent_schema, agent_entity)
        
        for key in results.keys():
            name = ds+"_"+key
            all_outputs[name] = results[key]

        with open("output_concise_1.json", "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=2)
        
        finished_names.append(ds)
        with open("finished_names.json", "w") as f:
            json.dump(finished_names, f)
        # break

    print(f"✔︎ Completed. Results written to output_concise.json")


if __name__ == "__main__":
    main()
