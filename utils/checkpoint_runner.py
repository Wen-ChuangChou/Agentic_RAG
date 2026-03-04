"""
General-purpose checkpoint runner for LLM evaluation.

Supports both agentic RAG and vanilla LLM testing with automatic
checkpoint/resume on API failures or interruptions.

Usage:
    from utils.checkpoint_runner import run_with_checkpoint

    # Agentic RAG
    def agentic_answer(question):
        enhanced = prompt_config["prompt"].format(question=question)
        return agent.run(enhanced)

    results = run_with_checkpoint(
        eval_dataset, agentic_answer,
        checkpoint_file="checkpoints/agentic_rag_qwen35.json",
        model_name="Qwen3.5 122B",
        prompt_name="guide_agent_system_prompt",
    )

    # Vanilla LLM
    def vanilla_answer(question):
        return llm.complete(question)

    results = run_with_checkpoint(
        eval_dataset, vanilla_answer,
        checkpoint_file="checkpoints/vanilla_qwen35.json",
        model_name="Qwen3.5 122B",
        prompt_name="none",
    )
"""

import json
import os
import time
from datetime import datetime
from typing import Callable, List, Optional

from tqdm import tqdm


def save_checkpoint(checkpoint_file: str, results: list, next_idx: int,
                    model_name: str = "unknown",
                    prompt_name: str = "unknown") -> None:
    """Save checkpoint data to file using atomic write to prevent corruption."""
    checkpoint_data = {
        "model_name": model_name,
        "prompt_name": prompt_name,
        "results": results,
        "next_idx": next_idx,
        "timestamp": datetime.now().isoformat(),
    }

    # Ensure the directory exists
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    # Atomic write: write to temp file first, then rename
    temp_file = f"{checkpoint_file}.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

    # Safely replace the old checkpoint file
    os.replace(temp_file, checkpoint_file)


def load_checkpoint(checkpoint_file: str) -> dict:
    """Load checkpoint data from file. Returns empty dict if not found."""
    if not os.path.exists(checkpoint_file):
        return {}

    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Error loading checkpoint: {e}")
        print("Starting from beginning.")
        return {}


def run_with_checkpoint(
    eval_dataset,
    answer_fn: Callable[[str], str],
    checkpoint_file: str = "checkpoints/checkpoint.json",
    model_name: str = "unknown",
    prompt_name: str = "unknown",
    delay: float = 0,
) -> List[dict]:
    """
    Run LLM evaluation with checkpointing to allow resuming from
    interruptions (API errors, rate limits, keyboard interrupt, etc.).

    Args:
        eval_dataset: HuggingFace dataset with "question", "answer",
                      and "source_doc" columns.
        answer_fn: A callable that takes a question string and returns
                   an answer string. This is the core abstraction that
                   makes this function work for any backend (agentic RAG,
                   vanilla LLM, etc.).
        checkpoint_file: Path to JSON file for storing checkpoint data.
        model_name: Name of the LLM model (for metadata tracking).
        prompt_name: Name of the prompt config used (for metadata tracking).
        delay: Seconds to wait between API calls (rate limiting).

    Returns:
        List of result dicts, each containing:
        {question, true_answer, source_doc, generated_answer}
    """
    results = []
    start_idx = 0

    # Load existing checkpoint if available
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        results = checkpoint.get("results", [])
        start_idx = checkpoint.get("next_idx", 0)
        prev_model = checkpoint.get("model_name", "unknown")
        prev_prompt = checkpoint.get("prompt_name", "unknown")
        print(
            f"Resuming from checkpoint at index {start_idx}/{len(eval_dataset)} "
            f"({len(results)} results already processed)"
        )
        print(f"  Previous run: model={prev_model}, prompt={prev_prompt}")

    if start_idx >= len(eval_dataset):
        print("All questions already processed!")
        return results

    total = len(eval_dataset)
    pbar = tqdm(total=total, initial=start_idx, desc="Evaluating")

    try:
        for idx in range(start_idx, total):
            example = eval_dataset[idx]
            question = example["question"]

            try:
                answer = answer_fn(question)

                print("=======================================================")
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                print(f'True answer: {example["answer"]}')

                result = {
                    "question": question,
                    "true_answer": example["answer"],
                    "source_doc": example["source_doc"],
                    "generated_answer": str(answer),
                }
                results.append(result)

                # Save checkpoint after each successful question
                save_checkpoint(
                    checkpoint_file, results, idx + 1,
                    model_name, prompt_name
                )
                pbar.update(1)

                # Optional rate limiting
                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                print(f"\nError at question {idx}: {e}")
                save_checkpoint(
                    checkpoint_file, results, idx,
                    model_name, prompt_name
                )
                print(f"Checkpoint saved. Re-run to resume from index {idx}.")
                raise

    except KeyboardInterrupt:
        print(f"\nInterrupted by user at question {idx}.")
        save_checkpoint(
            checkpoint_file, results, idx,
            model_name, prompt_name
        )
        print(f"Checkpoint saved. Re-run to resume from index {idx}.")

    finally:
        pbar.close()

    return results
