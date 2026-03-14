import os
import datasets
import json
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

load_dotenv()
dataset_name = "m-ric/huggingface_doc"
RESULTS_DIR = Path("results")
CHECKPOINTS_DIR = Path("checkpoints")

# Configurable chunk sizes
config = {
    "batch_size": 50,
    "max_workers": 4,
    "doc_chunk_size": 100,  # Parallel processing batch size
    "text_chunk_size": 200,  # Document content chunk size (tokens) 
    "text_chunk_overlap": 40,  # Overlap between chunks
    "force_rebuild": False,
    "use_parallel": True
}

##### Load or create vector database parallelly
vectordb = load_or_create_vectordb(dataset_name, **config)
eval_dataset = datasets.load_dataset("m-ric/huggingface_doc_qa_eval",
                                     split="train")

##### LLM model initialization

# model_name = "gemini-2.5-flash-preview-05-20"
# # model_name = "gemini-1.5-flash"
# answer_llm = OpenAIServerModel(
#     model_id=model_name,
#     api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
#     api_key=os.getenv("Gemini_API_KEY"),
#     temperature=0.2)

API_KEY = os.getenv("Blablador_API_KEY")
LLM = BlabladorChatModel(api_key=API_KEY)
model_name = "Qwen3.5 122B"  #Qwen3.5 122B, MiniMax-M2.5, NVIDIA-Nemotron
model_fullname = LLM.get_model_fullname(model_name)
print(f"The agentic RAG uses the following model: {model_fullname}\n")

TEMPERATURE = 0.2
answer_llm = OpenAIServerModel(
    model_id=model_fullname,
    api_base="https://api.helmholtz-blablador.fz-juelich.de/v1",
    api_key=API_KEY,
    max_tokens=16384,
    # flatten_messages_as_text=True,
    temperature=TEMPERATURE)

retriever_tool = RetrieverTool(vectordb)
agent = CodeAgent(
    tools=[retriever_tool],
    model=answer_llm,
    planning_interval=5,
    max_steps=30,
    verbosity_level=LogLevel.ERROR,
)

results_filename = f"{model_name}_vect{config['text_chunk_size']}_t{TEMPERATURE}.json"

##### Using agentic RAG to answer questions


# Define answer function using agent + YAML prompt
def agentic_answer(question):
    enhanced = prompt_config["prompt"].format(question=question)
    return agent.run(enhanced)


# load system prompt for using agentic RAG
prompt_filename = "guide_agent_system_prompt.yaml"
prompt_path = Path("prompts") / prompt_filename
with open(prompt_path, "r", encoding="utf-8") as f:
    prompt_config = yaml.safe_load(f)

print(f"Evaluating agentic RAG performance using model: {model_fullname}\n")

llm_outputs = {}
llm_outputs["agentic_rag"] = run_with_checkpoint(
    eval_dataset,
    agentic_answer,
    checkpoint_file=CHECKPOINTS_DIR / f"{model_name}_agentic_rag.json",
    model_name=model_fullname,
    prompt_name="guide_agent_system_prompt",
)

# save_results(CHECKPOINTS_DIR / results_filename, "agentic_rag",
#              llm_outputs["agentic_rag"])
print(f"\n{'='*50}")
print("Agentic RAG evaluation completed.")
print(f"{'='*50}\n")

##### using RAG only to answer questions


def rag_answer(question):
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


print(f"Evaluating standard RAG performance using model: {model_fullname}\n")
llm_outputs["standard_rag"] = run_with_checkpoint(
    eval_dataset,
    rag_answer,
    checkpoint_file=CHECKPOINTS_DIR / f"{model_name}_rag.json",
    model_name=model_fullname,
    prompt_name="none",
)

# save_results(CHECKPOINTS_DIR / results_filename, "standard_rag",
#              llm_outputs["standard_rag"])
print(f"\n{'='*50}")
print("RAG evaluation completed.")
print(f"{'='*50}\n")

##### using vanilla LLM to answer questions


def vanilla_answer(question):
    prompt = f"""Answer the following question as clearly and concisely as possible.
Use your own knowledge to respond.
If the question is ambiguous or cannot be answered definitively, state so clearly.

Question:
{question}

"""
    messages = [{"role": "user", "content": prompt}]
    return answer_llm.generate(messages).content


print(f"Utilizing {model_fullname} for question answering...\n")

llm_outputs["standard"] = run_with_checkpoint(
    eval_dataset,
    vanilla_answer,
    checkpoint_file=CHECKPOINTS_DIR / f"{model_name}_vallina.json",
    model_name=model_fullname,
    prompt_name="none",
)

# save_results(CHECKPOINTS_DIR / results_filename, "standard", llm_outputs["standard"])
print(f"\n{'='*50}")
print("Vanilla LLM evaluation completed.")
print(f"{'='*50}\n")

prompt_filename = "evaluation_prompt.yaml"
prompt_path = Path("prompts") / prompt_filename
with open(prompt_path, "r", encoding="utf-8") as f:
    evaluation_prompt = yaml.safe_load(f)

##### Evaluation of 3 different LLM systems
eval_model_name = "GPT-OSS-120b"
eval_model_fullname = LLM.get_model_fullname(eval_model_name)
print(f"Performance Evaluation Model: {eval_model_fullname}\n")

evaluation_llm = OpenAIServerModel(
    model_id=eval_model_fullname,
    api_base="https://api.helmholtz-blablador.fz-juelich.de/v1",
    api_key=API_KEY,
    max_tokens=16384,
    temperature=0)

# # evaluation_llm_name = "gemini-2.0-flash"
# evaluation_llm_name = "gemini-3.1-flash-lite-preview"
# evaluation_llm = OpenAIServerModel(
#     model_id=evaluation_llm_name,
#     api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
#     api_key=os.getenv("Gemini_API_KEY"),
#     temperature=0)

evaluated = run_evaluation_with_checkpoint(
    llm_outputs,
    evaluation_prompt["prompt"],
    evaluation_llm,
    checkpoint_file=CHECKPOINTS_DIR / f"{model_name}_eval.json",
    delay=5,
    max_retries=5,
    max_consecutive_errors=3,  # stop after 3 consecutive failures
)
# Convert to DataFrames for scoring
results = {}
for system_type, outputs in evaluated.items():
    results[system_type] = pd.DataFrame.from_dict(outputs)
    results[system_type] = results[
        system_type].loc[~results[system_type]["generated_answer"].str.
                         contains("Error", na=False)]

##### Show the accuracy of 3 different systems
DEFAULT_SCORE = 2  # Give average score whenever scoring fails


def fill_score(x):
    try:
        return int(x)
    except:
        return DEFAULT_SCORE


for system_type in [
        "agentic_rag",
        "standard_rag",
        "standard",
]:

    results[system_type]["eval_score_LLM_judge_int"] = (
        results[system_type]["eval_score_LLM_judge"].fillna(
            DEFAULT_SCORE).apply(fill_score))
    results[system_type]["eval_score_LLM_judge_int"] = (
        results[system_type]["eval_score_LLM_judge_int"] - 1) / 2

    print(
        f"Average score for {system_type} : {results[system_type]['eval_score_LLM_judge_int'].mean()*100:.1f}%"
    )

print(f"{'='*50}\n")

##### save results after evaluation of LLM system performance
meta_data = {
    "model_name": model_name,
    "model_id": model_fullname,
    "prompt_filename": prompt_filename,
    "eval_model_name": eval_model_name,
    "eval_model_id": eval_model_fullname,
}

eval_performance_filename = f"{model_name}_vect{config['text_chunk_size']}_t{TEMPERATURE}.json"
save_evaluation_results(meta_data, results, RESULTS_DIR,
                        eval_performance_filename)
