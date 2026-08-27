# Aster & Row Support Agent — System Specifications

This document outlines the structural design constraints and validation mechanisms built into the support agent.

## 1. RAG Index & Vector Space
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
* **Metric**: L2-normalized cosine similarity
* **Storage Format**: Flat JSON file index
* **Noise Filter**: Irrelevant RAG results below `0.33` score are dropped, triggering safe abstention.

## 2. Privacy & Data Handling
* **Restricted Fields**: Customer email, address, internal warehouse notes, and risk scores are filtered at the database interface.
* **Order Status Trimming**: Cancelled or returned orders have tracking details and ETAs suppressed before the payload reaches the LLM context.

## 3. Human Handoff triggers
Handoff warnings are recommended under three specific conditions:
1. Operational mutations (requests for cancellation, address changes, or refunds on shipped orders).
2. Product care instructions conflict (like the Breeze Tumbler dishwasher conflict).
3. Access attempts on internal-only data or requests for legacy policies.
