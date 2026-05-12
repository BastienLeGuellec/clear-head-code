"""
generate_summaries.py — LLM-based lay summary generation
=========================================================
Reads brain MRI reports from an Excel file, generates patient-centered
lay summaries using a locally hosted model via Ollama, and saves the
results back to an Excel file.

The prompt is a zero-shot instruction previously validated by three
neuroradiologists (see Supplementary Material).

Model    : Mistral Small 3.2 (mistral-small3.2:24b) via Ollama
Endpoint : http://localhost:11434  (Ollama local server)
Settings : temperature=0 (deterministic), top_p=0.9, seed=42

Usage:
    # 1. Pull the model once:
    #    ollama pull mistral-small3.2:24b
    # 2. Run the script:
    python generate_summaries.py

Dependencies: pandas, openai (>=1.0), openpyxl
"""

import pandas as pd
from openai import OpenAI

# ── Ollama client (OpenAI-compatible local endpoint) ─────────────────────────
# Ollama exposes an OpenAI-compatible REST API at http://localhost:11434/v1.
# No real API key is required; "ollama" is passed as a placeholder.
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

# ── Generation parameters ─────────────────────────────────────────────────────
MODEL       = "mistral-small3.2:24b"   # ollama pull mistral-small3.2:24b
TEMPERATURE = 0                         # deterministic output
TOP_P       = 0.9
SEED        = 42

SYSTEM_PROMPT = (
    "You are a helpful assistant that explains radiology reports to patients "
    "in French. The summary should be clear, concise, and easy to understand "
    "for someone without a medical background."
)

USER_PROMPT_TEMPLATE = (
    "Please summarize the following medical report to a patient in French, "
    "in a few sentences. Begin by contextualizing the MRI exam, then explain "
    "the main findings, using clear localisation when needed, synonyms or "
    "definitions of complicated words, then a sentence on the relationship "
    "between the main findings and the symptoms. "
    "Start your summary with: 'Résumé patient:' "
    "Here is the report:\n\n{report_text}"
)


def summarize_report(report_text: str) -> dict:
    """Generate a patient-centered lay summary for a single MRI report.

    Args:
        report_text: Full text of the radiology report.

    Returns:
        Dictionary with the summary and generation metadata.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(
                report_text=report_text)},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        seed=SEED,
    )

    return {
        "summary":     response.choices[0].message.content,
        "model":       MODEL,
        "temperature": TEMPERATURE,
        "top_p":       TOP_P,
        "seed":        SEED,
    }


def process_reports(input_path: str, output_path: str) -> None:
    """Read reports from Excel, generate summaries, and save results.

    Args:
        input_path:  Path to the input Excel file (must contain a 'report' column).
        output_path: Path where the enriched Excel file will be saved.
    """
    try:
        df = pd.read_excel(input_path, sheet_name='Sheet1')
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found.")
        return

    if 'report' not in df.columns:
        print("Error: the Excel file must contain a column named 'report'.")
        return

    # Initialise output columns
    for col in ['summary', 'model', 'temperature', 'top_p', 'seed']:
        df[col] = ""

    for idx, row in df.iterrows():
        report_text = row['report']
        if pd.isna(report_text):
            continue
        print(f"Processing report {idx + 1} / {len(df)}...")
        try:
            result = summarize_report(report_text)
            for col in ['summary', 'model', 'temperature', 'top_p', 'seed']:
                df.at[idx, col] = result[col]
        except Exception as exc:
            print(f"  ⚠ Report {idx + 1} failed: {exc}")

    df.to_excel(output_path, index=False)
    print(f"\nDone. Results saved to '{output_path}'.")


if __name__ == "__main__":
    process_reports(
        input_path  = 'selected_reports.xlsx',
        output_path = 'selected_reports_with_summaries.xlsx',
    )
