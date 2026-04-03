# HyperGraphReasoning for Logistics Automation

This repository is a logistics-focused adaptation of the methodology introduced in:

**"Higher-Order Knowledge Representations for Agentic Scientific Reasoning"**  
Isabella Stewart, Markus J. Buehler (MIT, 2026)

We are repurposing that higher-order knowledge representation framework from scientific text to logistics operations data (shipment events, route constraints, carrier actions, SLA/KPI violations, etc.).

---

## What this project does

This pipeline converts logistics documents into:

- Directed knowledge graphs (pairwise relations)
- Hypergraphs (higher-order multi-entity events)
- Graph/hypergraph artifacts for analysis and agent workflows

Typical extracted entities include:

- shipment, order, SKU, package, pallet, container
- route, lane, stop, facility, warehouse, dock
- carrier, vehicle, customer, supplier
- SLA, KPI, delay, exception, capacity, inventory state

---

## Acknowledgement of repurposing

This codebase is based on and inspired by the original hypergraph reasoning framework from the paper above.  
Our use case shifts the domain from scientific materials reasoning to logistics automation and operational intelligence.

Core algorithmic ideas retained:

- chunked document processing
- LLM-based relation extraction
- higher-order event representation via hyperedges
- graph + hypergraph analysis workflows

Main adaptations in this repository:

- logistics-specific extraction prompts
- JSON-structured output handling (no Instructor requirement)
- logistics-oriented graph semantics and examples

---

## Repository layout

- `GraphReasoning/` — main source package
- `build/lib/GraphReasoning/` — build copy of package files
- `Notebooks/SG/` — notebooks for generation, analysis, and agents
- `run_make_new_hypergraph.py` — main script to build integrated outputs
- `artifacts/`
  - `sg/graphs/` — per-document graph artifacts
  - `sg/integrated/` — merged/integrated graph artifacts
  - `cache/chunks/` — chunk-level cache

---

## Quickstart (logistics workflow)

### 1) Create environment

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2) Configure model access

Set required environment variables in your shell or `.env`:

- `OPENAI_API_KEY` (or your provider key)
- `MODEL_NAME`
- `URL` (if applicable to your provider/client wrapper)

### 3) Put logistics markdown files here

`Notebooks/SG/Data/`

### 4) Run pipeline

```bash
python run_make_new_hypergraph.py
```

### 5) Outputs

- per-doc outputs in `artifacts/sg/graphs/`
- integrated outputs in `artifacts/sg/integrated/`

---

## Structured output contract (JSON)

The extraction now supports plain JSON outputs directly.

### Directed graph extraction expects:

```json
{
  "nodes": [{"id": "...", "type": "..."}],
  "edges": [{"source": "...", "target": "...", "relation": "..."}]
}
```

### Hypergraph extraction expects:

```json
{
  "events": [
    {
      "source": ["..."],
      "relation": "...",
      "target": ["..."]
    }
  ]
}
```

Each event maps to one hyperedge over `source ∪ target`.

---

## Optional PDF export dependency

If you use `save_PDF=True`, install:

```bash
pip install pdfkit
```

and ensure `wkhtmltopdf` is installed and on PATH.

---

## Suggested domain onboarding checklist

Before production usage for logistics:

1. Curate a validation corpus (orders, shipment updates, SOPs, incident reports)
2. Define canonical relation vocabulary (e.g., `departs_from`, `arrives_at`, `delayed_by`)
3. Evaluate extraction precision/recall on a hand-labeled set
4. Add relation normalization and schema validation retries
5. Freeze prompt versioning + cache versioning for reproducibility

---

## Citation (original method)

If this work contributes to your research or systems documentation, cite the original paper:

```bibtex
@article{stewartbuehler2025hypergraphreasoning,
  title     = {Higher-Order Knowledge Representations for Agentic Scientific Reasoning},
  author    = {I.A. Stewart and M.J. Buehler},
  journal   = {arXiv},
  year      = {2026},
  doi       = {https://arxiv.org/abs/2601.04878}
}
```
