# ResolveGPT  
### AI-Assisted Software Support Ticket Resolution System using Retrieval Augmented Generation (RAG)

---

## Overview

ResolveGPT is an AI-powered system built to assist software support teams in resolving customer and internal technical tickets more efficiently. The system analyzes previously resolved software support tickets and uses that historical knowledge to suggest accurate and context-aware resolutions for new incoming tickets.

In real-world software organizations, support teams frequently deal with repetitive issues such as login failures, API errors, configuration problems, and deployment issues. Although solutions for these problems often already exist in past tickets, agents typically spend a significant amount of time manually searching through old records, documentation, or internal tools to find the correct resolution. This process is inefficient and heavily dependent on individual experience.

ResolveGPT addresses this challenge by combining semantic search and generative AI, allowing the system to understand the meaning of a new ticket, retrieve similar past cases, and generate a suggested resolution based on real historical data. The final decision is always left to the human support agent, ensuring safety and reliability.

---

## Problem Statement

Traditional ticket search systems rely on keyword matching, which often fails to capture the actual intent or context of a support issue. Two tickets describing the same problem may use completely different wording, making keyword-based search unreliable. As a result, valuable organizational knowledge stored in past tickets is underutilized.

The objective of this project is to build a system that can perform meaning-based retrieval of support tickets and assist agents by providing grounded, explainable, and reusable resolution suggestions. This is achieved using a Retrieval Augmented Generation (RAG) architecture, which is widely adopted in modern enterprise-grade GenAI systems.

---

## How the System Works

The system follows a structured, production-oriented pipeline.

First, a dataset containing historical software support tickets is prepared. Each ticket includes an issue description along with the resolution provided by a human agent. These ticket descriptions are converted into numerical vector representations known as embeddings. Embeddings capture the semantic meaning of text, allowing the system to compare tickets based on intent rather than keywords.

All embeddings are stored in a FAISS vector index. FAISS is used because it enables fast and efficient similarity search, even when working with large datasets. This makes the system scalable and suitable for real-world usage.

When a new support ticket is received, the system generates an embedding for the new ticket using the same embedding model. This embedding is compared against the stored vectors in FAISS to retrieve the most similar previously resolved tickets. The retrieved ticket descriptions and resolutions provide factual context relevant to the current issue.

This retrieved context is then passed to a Large Language Model (LLM). Instead of generating answers blindly, the LLM uses the retrieved historical information to generate a resolution suggestion that is accurate, relevant, and grounded in real data. This approach significantly reduces the risk of hallucinated or incorrect responses.

The generated resolution is finally presented to a human support agent, who can review, modify, or directly use the suggestion before responding to the user. This human-in-the-loop design ensures trust, accountability, and enterprise readiness.

---

## Why Retrieval Augmented Generation (RAG)

Using a standalone language model for support automation can be unreliable, as language models may produce confident but incorrect answers. Retrieval Augmented Generation mitigates this risk by forcing the model to rely on retrieved factual context rather than assumptions.

In ResolveGPT, RAG ensures that every generated resolution is based on real, previously solved tickets. This improves accuracy, transparency, and alignment with real-world support workflows.

---

## Technologies Used

Python is used as the core programming language due to its strong ecosystem for machine learning, natural language processing, and backend development.

FAISS is used as the vector database to store and retrieve ticket embeddings efficiently. It enables high-performance similarity search and is widely adopted in production systems.

Large Language Models are used to generate natural-language resolution suggestions. The model is guided using retrieved historical context, ensuring that responses remain relevant and grounded.

FastAPI is used to expose the system as an API. This allows the solution to be easily integrated with dashboards, internal tools, or external applications. FastAPI was chosen for its performance, simplicity, and clean API design.

---
## Project Structure

ResolveGPT/
├── app/
│ ├── rag_pipeline.py
│ ├── llm_client.py
│ ├── embedder.py
│ ├── faiss.py
│ └── main.py
│
├── templates/
│ └── index.html
│
├── data/
│ └── software_tickets.csv
│
├── requirements.txt
└── README.md

---

## Example Workflow

A new support ticket reports that a user is unable to log in after resetting their password and encounters an authentication error. The system processes the ticket, retrieves similar past authentication-related issues, and uses those examples to generate a resolution explaining the likely cause and recommended steps to fix the issue.

The support agent receives a clear, context-aware suggestion instead of starting from scratch, which significantly reduces resolution time and improves consistency.

---

## What This Project Demonstrates

This project demonstrates practical knowledge of modern Generative AI system design, including semantic search, vector databases, retrieval-augmented generation, and API-driven architectures. It reflects real-world problem-solving rather than experimental or toy use cases.

---
