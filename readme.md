# **Agentic Retrieval-Augmented-Generation (RAG): AI Agent for self-query and query reformulation**

## **Project Objective**

To implement and evaluate Agentic Retrieval Augmented Generation (RAG), comparing its performance against traditional RAG and standalone Large Language Models (LLMs) in answering technical questions related to Hugging Face ecosystem packages.

# **Motivation**

The primary motivation behind this project is to explore and demonstrate the advanced capabilities of RAG by incorporating intelligent agents.  
Traditional RAG systems, while powerful, often follow a fixed retrieve-then-generate pattern. This project aims to move beyond that by showcasing how an agent-based approach can introduce more dynamic decision-making, iterative refinement, and tool utilization into the RAG pipeline. By implementing an agentic framework, we seek to address the limitations of basic RAG, such as handling complex, multi-step queries or adapting to diverse information retrieval needs.  
The implementation focuses on creating an agent that can intelligently interact with external knowledge sources, evaluate retrieved content, and refine its approach based on the outcome, ultimately leading to more accurate, robust, and contextually rich responses from the Large Language Model.

## **Key Capabilities of This Agentic RAG Implementation**

* **Query Strategy & Refinement:** Strategically determines and combines keywords for search queries, iteratively refining them based on retrieval results to optimize relevance and coverage.  
* **Iterative Query Refinement:** If initial retrieval is insufficient, an agent can reformulate queries or increase the number of documents it can retrieve.  
* **Document Evaluation:** Assess the relevance and quality of retrieved information to answer the question.  
* **Multi-step Reasoning:** Use retrieved information to answer questions by chaining together multiple retrieval and generation steps.  
* **Self-Correction & Backtracking:** If a generated answer is unsatisfactory, an agent can devise new strategies to try a different approach.

Agentic RAG significantly enhances the RAG pipeline, providing more sophisticated reasoning, planning, and execution capabilities for robustly handling complex information-seeking tasks. This repository leverages the smolagent package to build the underlying agentic framework.

## **Parallel Vector Database Creation with Optimized Document Processing**

This implementation employs an **optimized parallel processing approach** for creating vector databases from extensive document collections. For this project, it specifically utilizes the database, which contains information on packages developed by Hugging Face. This technique integrates several performance optimization strategies:

### **Key Features**

* **Parallel Document Splitting**: Documents are processed concurrently using ThreadPoolExecutor, splitting the workload across multiple threads to significantly reduce processing time for large datasets.  
* **Batch Embedding**: Instead of embedding documents one at a time, the system processes documents in configurable batches (default 100), creating initial FAISS vectorstores and then merging them incrementally to manage memory usage efficiently.  
* **Thread-Safe Processing**: Uses a custom DocumentProcessor class with threading locks to ensure safe concurrent access to the text splitter, preventing race conditions during parallel execution.  
* **Intelligent Fallback**: Automatically falls back to sequential processing if parallel execution fails, ensuring robustness across different environments and dataset sizes.  
* **Deduplication**: Removes duplicate documents based on content hash to optimize storage and retrieval performance.  
* **Persistent Caching**: Saves and loads pre-built vector databases to disk, avoiding expensive recomputation on subsequent runs unless explicitly forced to rebuild.

### **Technical Implementation**

* Uses HuggingFace's gte-small model for embeddings with tokenizer-aware text splitting  
* Implements FAISS with cosine distance for efficient similarity search  
* Configurable chunk sizes (tokens) and overlap for optimal retrieval granularity  
* Progress tracking with tqdm for long-running operations  
* Sanitized file naming for cross-platform compatibility

This approach is particularly effective for large-scale RAG applications where document preprocessing time is a bottleneck, providing significant speedup while maintaining retrieval quality.

## **Results**

Performance was evaluated using the [Hugging Face technical Q\&A dataset](https://huggingface.co/datasets/m-ric/huggingface_doc_qa_eval) with Gemini 1.5, Gemini 2.0, and Gemini 2.5 LLMs. Agentic RAG consistently demonstrated superior accuracy compared to Standard RAG across all models. The relative performance trends among the different LLMs remained consistent between Agentic RAG and Standard RAG implementations. As expected, standalone LLM models generally exhibited lower accuracy, with Gemini 1.5 showing the most significant performance deficit. Gemini 2.0 was specifically utilized to evaluate the consistency of generated answers against the ground truth in the [Hugging Face technical Q\&A dataset](https://huggingface.co/datasets/m-ric/huggingface_doc_qa_eval).


<div align="center">

| Model | Agentic RAG Accuracy | Standard RAG Accuracy | LLM Only Accuracy |
|:---------------------:|:--------------------:|:----------------------:|:------------------:|
| Gemini-1.5-flash      |        91.5%         |         85.4%          |       35.4%        |
| Gemini-2.0-flash      |        90.8%         |         85.4%          |       64.1%        |
| Gemini-2.5-flash-preview-05-20 |        90.8%         |         86.2%          |       63.8%        |

</div>

## **Improvement**

Future improvements for this project include:

1. Broaden LLM Evaluation: Expand testing to include a wider variety of LLMs or different agentic RAG architectural patterns (e.g., integrating various tools, multi-agent systems) to assess generalizability and identify optimal configurations.  
2. Refine Agent Prompting: Enhance the system prompts to more precisely guide agent behavior, leading to increased efficiency and better alignment with desired task execution.  
3. Enhance Objective Evaluation Criteria: Develop more rigorous system prompts for evaluating LLM responses, ensuring objectivity, especially concerning conciseness and directness. Responses that are overly verbose or contain extraneous information, even if partially correct, should be scored down appropriately.

## **Reference:**

This repository is extended from the work of the [Hugging Face Agentic RAG Cookbook](https://huggingface.co/learn/cookbook/agent_rag).