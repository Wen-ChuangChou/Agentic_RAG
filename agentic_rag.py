"""
This script evaluates and compares three different Question-Answering (QA) systems:
1. Agentic RAG: Uses a Code Agent with access to a retriever tool.
2. Standard RAG: Uses a fixed-retrieval prompt with an LLM.
3. Vanilla LLM: Answers questions using the LLM's internal knowledge only.

The evaluation is performed using a Hugging Face dataset and an LLM-as-judge for scoring.
It uses checkpointing to allow resuming long-running evaluations.
"""
import os
import time
import datasets
import pandas as pd
import yaml
from dotenv import load_dotenv
from pathlib import Path
from smolagents import OpenAIServerModel, CodeAgent
from smolagents.monitoring import LogLevel
from utils.agent_tools import RetrieverTool
from utils.blablador_helper import BlabladorChatModel
from utils.checkpoint_runner import run_with_checkpoint, run_evaluation_with_checkpoint
from utils.results_manager import save_evaluation_results
from utils.vectordb_utils import load_or_create_vectordb


def agentic_answer(question, agent, prompt_config):
    """
    Generate an answer using a CodeAgent with a tailored system prompt.
    
    Args:
        question (str): The user's query.
        agent (CodeAgent): The smolagents agent instance.
        prompt_config (dict): Configuration containing the system prompt template.
    
    Returns:
        str: The generated answer from the agent.
    """
    enhanced = prompt_config["prompt"].format(question=question)
    return agent.run(enhanced)


def rag_answer(question, retriever_tool, answer_llm):
    """
    Generate an answer using standard RAG (Retrieval-Augmented Generation).
    
    Args:
        question (str): The user's query.
        retriever_tool (RetrieverTool): Tool for fetching relevant documents.
        answer_llm (OpenAIServerModel): The LLM used for answering.
    
    Returns:
        str: The generated answer based on retrieved context.
    """
    context = retriever_tool(question, k=5)
    prompt = f"""Given the question and supporting documents below, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
If the question is ambiguous or cannot be answered definitively, state so clearly.

Question:
{question}

{context}
"""
    messages = [{"role": "user", "content": prompt}]
    return answer_llm.generate(messages).content


def vanilla_answer(question, answer_llm):
    """
    Generate an answer using only the LLM's pre-trained knowledge.
    
    Args:
        question (str): The user's query.
        answer_llm (OpenAIServerModel): The LLM used for answering.
    
    Returns:
        str: The generated answer.
    """
    prompt = f"""Answer the following question as clearly and concisely as possible.
Use your own knowledge to respond.
If the question is ambiguous or cannot be answered definitively, state so clearly.

Question:
{question}

"""
    messages = [{"role": "user", "content": prompt}]
    return answer_llm.generate(messages).content


def fill_score(x, default_score):
    """
    Convert a string score to an integer, falling back to a default if parsing fails.
    
    Args:
        x (any): The score value to convert.
        default_score (int): The fallback value.
    
    Returns:
        int: The converted score.
    """
    try:
        return int(x)
    except:
        return default_score


def main():
    # Initialize environment and paths
    load_dotenv()
    dataset_name = "m-ric/huggingface_doc"
    RESULTS_DIR = Path("results")
    CHECKPOINTS_DIR = Path("checkpoints")

    # Configurable parameters for vector DB and processing
    config = {
        "batch_size": 50,
        "max_workers": 4,
        "doc_chunk_size": 100,  # Parallel processing batch size
        "text_chunk_size": 200,  # Document content chunk size (tokens) 
        "text_chunk_overlap": 40,  # Overlap between chunks
        "force_rebuild": False,
        "use_parallel": True
    }

    # Load or create vector database for retrieval
    vectordb = load_or_create_vectordb(dataset_name, **config)
    eval_dataset = datasets.load_dataset("m-ric/huggingface_doc_qa_eval",
                                         split="train")

    # Initialize LLM via Blablador
    API_KEY = os.getenv("Blablador_API_KEY")
    LLM_helper = BlabladorChatModel(api_key=API_KEY)
    model_name = "Qwen3.6-35B-A3B"  # Options: Qwen3.5-122B, MiniMax-M2.7
    model_fullname = LLM_helper.get_model_fullname(model_name)
    print(f"The agentic RAG uses the following model: {model_fullname}\n")

    TEMPERATURE = 0.2
    answer_llm = OpenAIServerModel(
        model_id=model_fullname,
        api_base="https://api.helmholtz-blablador.fz-juelich.de/v1",
        api_key=API_KEY,
        max_tokens=16384,
        temperature=TEMPERATURE)

    # Setup Agentic RAG components
    retriever_tool = RetrieverTool(vectordb)
    agent = CodeAgent(
        tools=[retriever_tool],
        model=answer_llm,
        planning_interval=5,
        max_steps=30,
        verbosity_level=LogLevel.ERROR,
    )

    # Load custom system prompt for the agent
    prompt_filename = "guide_agent_system_prompt.yaml"
    prompt_path = Path("prompts") / prompt_filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)

    print(
        f"Evaluating agentic RAG performance using model: {model_fullname}\n")

    timing = {}

    # 1. Evaluate Agentic RAG
    llm_outputs = {}
    llm_outputs["agentic_rag"], timing[
        "agentic_rag_seconds"] = run_with_checkpoint(
            eval_dataset,
            lambda q: agentic_answer(q, agent, prompt_config),
            checkpoint_file=CHECKPOINTS_DIR / f"{model_name}_agentic_rag.json",
            model_name=model_fullname,
            prompt_name="guide_agent_system_prompt",
        )

    print(f"\n{'='*50}")
    print(
        f"Agentic RAG evaluation completed in {timing['agentic_rag_seconds']}s."
    )
    print(f"{'='*50}\n")

    # 2. Evaluate Standard RAG
    print(
        f"Evaluating standard RAG performance using model: {model_fullname}\n")
    llm_outputs["standard_rag"], timing[
        "standard_rag_seconds"] = run_with_checkpoint(
            eval_dataset,
            lambda q: rag_answer(q, retriever_tool, answer_llm),
            checkpoint_file=CHECKPOINTS_DIR / f"{model_name}_rag.json",
            model_name=model_fullname,
            prompt_name="none",
        )

    print(f"\n{'='*50}")
    print(f"RAG evaluation completed in {timing['standard_rag_seconds']}s.")
    print(f"{'='*50}\n")

    # 3. Evaluate Vanilla LLM
    print(f"Utilizing {model_fullname} for vanilla question answering...\n")
    llm_outputs["standard"], timing["vanilla_seconds"] = run_with_checkpoint(
        eval_dataset,
        lambda q: vanilla_answer(q, answer_llm),
        checkpoint_file=CHECKPOINTS_DIR / f"{model_name}_vallina.json",
        model_name=model_fullname,
        prompt_name="none",
    )

    print(f"\n{'='*50}")
    print(f"Vanilla LLM evaluation completed in {timing['vanilla_seconds']}s.")
    print(f"{'='*50}\n")

    # Setup LLM-as-judge for results evaluation
    eval_prompt_filename = "evaluation_prompt.yaml"
    eval_prompt_path = Path("prompts") / eval_prompt_filename
    with open(eval_prompt_path, "r", encoding="utf-8") as f:
        evaluation_prompt = yaml.safe_load(f)

    eval_model_name = "GPT-OSS-120b"
    eval_model_fullname = LLM_helper.get_model_fullname(eval_model_name)
    print(f"Performance Evaluation Model: {eval_model_fullname}\n")

    evaluation_llm = OpenAIServerModel(
        model_id=eval_model_fullname,
        api_base="https://api.helmholtz-blablador.fz-juelich.de/v1",
        api_key=API_KEY,
        max_tokens=16384,
        temperature=0)

    # Perform evaluation of all system outputs
    evaluated, timing["evaluation_seconds"] = run_evaluation_with_checkpoint(
        llm_outputs,
        evaluation_prompt["prompt"],
        evaluation_llm,
        checkpoint_file=CHECKPOINTS_DIR / f"{model_name}_eval.json",
        delay=5,
        max_retries=5,
        max_consecutive_errors=3,
    )
    timing["total_seconds"] = sum(timing.values())

    # Process evaluated results - keep ALL rows including [FAILED] placeholders
    results = {}
    for system_type, outputs in evaluated.items():
        results[system_type] = pd.DataFrame.from_dict(outputs)

    # Calculate and display accuracy scores
    DEFAULT_SCORE_VAL = 2  # Average score used when scoring fails
    scores = {}
    for system_type in ["agentic_rag", "standard_rag", "standard"]:
        if system_type not in results:
            print(f"Skipping '{system_type}': no evaluated results available.")
            continue
        df = results[system_type]
        # [FAILED] placeholders (API errors) get score=1 (worst) directly,
        # bypassing the LLM judge so failures count against the average.
        failed_mask = df["generated_answer"] == "[FAILED]"
        df.loc[failed_mask, "eval_score_LLM_judge"] = 1
        results[system_type] = df
        # Convert text scores to numeric and normalize
        results[system_type]["eval_score_LLM_judge_int"] = (
            results[system_type]["eval_score_LLM_judge"].fillna(
                DEFAULT_SCORE_VAL).apply(
                    lambda x: fill_score(x, DEFAULT_SCORE_VAL)))
        # Scale to percentage (assuming original scale is 1-3)
        results[system_type]["eval_score_LLM_judge_int"] = (
            results[system_type]["eval_score_LLM_judge_int"] - 1) / 2

        avg_score = results[system_type]['eval_score_LLM_judge_int'].mean(
        ) * 100
        scores[system_type] = avg_score
        print(f"Average score for {system_type} : {avg_score:.1f}%")

    print(f"{'='*50}\n")

    # Persist final evaluation results to disk
    meta_data = {
        "model_name": model_name,
        "model_id": model_fullname,
        "prompt_filename": prompt_filename,
        "eval_model_name": eval_model_name,
        "eval_model_id": eval_model_fullname,
        "timing": timing,
        "scores": scores,
    }
    eval_performance_filename = f"{model_name}_vect{config['text_chunk_size']}_t{TEMPERATURE}.json"
    save_evaluation_results(meta_data, results, RESULTS_DIR,
                            eval_performance_filename)


if __name__ == "__main__":
    main()
