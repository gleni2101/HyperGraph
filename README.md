# Higher-Order Knowledge Representations for Agentic Scientific Reasoning


#### Isabella Stewart, Markus J. Buehler, MIT, 2026 
Corresponding author: mbuehler@MIT.EDU

#### Laboratory for Atomistic and Molecular Mechanics (LAMM), Massachusetts Institute of Technology (MIT), Cambridge, MA 02139, USA

## Summary

Scientific inquiry increasingly requires systems-level reasoning that integrates heterogeneous experimental data, cross-domain knowledge, and mechanistic evidence into coherent, interpretable scientific explanations. While Large Language Models (LLMs) offer certain inferential capabilities for such tasks, they often depend on structured substrates such as knowledge graphs (KGs) for factual grounding. However, traditional pairwise KGs are ill-suited for scientific reasoning as they cannot adequately capture higher-order interactions among multiple entities that often govern emergent physical system behavior. To address this here we introduce a methodology for constructing hypergraph-based knowledge representations from large data corpora that move beyond traditional pairwise graphs. By explicitly encoding higher-order relationships, hypergraphs provide a more expressive and faithful framework for modeling the mechanistically coupled, multi-entity structures and physical behavior that underlie complex scientific phenomena. 

Applied to a scientific corpus on biocomposite scaffolds, our framework reveals that domain knowledge can be organized into a scale-free hypergraph with ego hubs around highly connected concepts. This representation enables an explicit detection of co-occurring ideas and inter-concept influences, clarifying how these relationships shape the emergent topology of the knowledge network. We further demonstrate that equipping agentic systems with hypergraph traversal tools enables them to exploit structural features, particularly hyperedge intersections, to ground beliefs and more effectively support mechanistic inference for hypothesis generation for the design of novel composite scaffold materials. This work establishes an agentic reasoning system using hypergraphs to capture higher-order relationships, enabling more effective use of large scientific datasets to accelerate scientific discovery.

Our contributions are:

1. A method for constructing hypergraphs from large-scale scientific text.
2. Techniques for analyzing hypergraphs derived from scientific corpora.
3. Agentic reasoning tools that operate over hypergraph representations.


## Getting Started

## Quickstart (run in this order)

This is the shortest path from raw markdown files to visible graph outputs.

### 0) Prepare your inputs
- Put your markdown files in: `Notebooks/SG/Data`
- Ensure your `.env` contains:
  - `URL`
  - `MODEL_NAME`
  - `OPENAI_API_KEY`

### 1) Create and activate environment

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install project

```bash
git clone https://github.com/lamm-mit/HyperGraphReasoning.git
cd HyperGraphReasoning
pip install -e .
```

### 3) Run the hypergraph pipeline

From repo root:

```bash
python run_make_new_hypergraph.py
```

This computes:
- per-document graphs
- merged integrated graphs
- node embeddings

### 4) See the computed graphs

Use either option below:

1. **Notebook visualization (recommended):**
   - Open `Notebooks/SG/make_new_hypergraph.ipynb`
   - Run the visualization cells at the end (interactive network + PCA embedding plot)

2. **Open exported interactive HTML directly:**
   - `artifacts/sg/integrated/latest_integrated_projection.html`

### 5) Where outputs are written

```bash
artifacts/
  sg/
    graphs/       # per-doc graph pickles + embeddings
    integrated/   # merged *_integrated.pkl + visualization html
  cache/
    chunks/       # chunk-level cache/temp artifacts
```

## 1) Instantiate Environment Configuration:

```
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2) Install CUDA (Optional)

Note: Ensure that the NVIDIA CUDA driver is already installed on your system before proceeding.
Install CUDA toolkit using your OS package manager or NVIDIA installer.

## 3) (Optional) Install llama-ccp-python
Required only if you intend to run models locally (≥ v0.2.0). 
You can skip this step if you’re using API-based LLMs.

```
CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=on" FORCE_CMAKE=1 pip install git+https://github.com/abetlen/llama-cpp-python@v0.3.8 --verbose
```


## 4) Install HyperGraphReasoning package
Clone the repository and proceed with the direct installation.
```
git clone https://github.com/lamm-mit/HyperGraphReasoning.git
cd HyperGraphReasoning
pip install .
```

See requirements.txt for the full list of packages seen in the approach described in our Methods section.

***

![Project Logo](agents.png)

***
# Generating a Hypergraph

## 1) PDF2Markdown
To convert PDF to markdown, follow instructions for creating environment at https://github.com/datalab-to/marker.git

It's best to make a clean environment for marker separate from this project environment just for markdown conversion.

```bash
python -m venv .venv-marker
# Windows (PowerShell)
.\.venv-marker\Scripts\Activate.ps1
# macOS/Linux
source .venv-marker/bin/activate
python -m pip install --upgrade pip
pip install marker-pdf
```

To convert multiple files at once: 
```bash
marker /path/to/input/folder
```
To convert multiple files on multiple GPUs: 
```bash
NUM_DEVICES=4 NUM_WORKERS=15 marker_chunk_convert ../pdf_in ../md_out
```

Final Composite Paper List.xlsx contains the list of papers from the corpus on 'biocomposite scaffold' that were used in this study. We do not provide the full text PDFs nor the markdown files in this repository. 


## 2) Generating Hypergraph and Hypergraph Embeddings from Markdown Files

Convert the Jupyter notebook to a Python script:
```bash
jupyter nbconvert --to script make_hypergraph.ipynb
```
This will create 

```bash
make_hypergraph.py
```

Run the generated script: 
```bash
python make_hypergraph.py
```

Or run the clean single-script pipeline (local BGE + merge flow):
```bash
python run_make_new_hypergraph.py
```

Default output layout is now centralized under:
```bash
artifacts/
  sg/
    graphs/       # per-document graph pickles + embeddings
    integrated/   # merged *_integrated.pkl files + visual exports
  cache/
    chunks/       # chunk-level temp/cache files
```

Example with explicit paths:
```bash
python run_make_new_hypergraph.py --doc-data-dir Notebooks/SG/Data --data-dir artifacts/sg/graphs --data-dir-output artifacts/sg/integrated --cache-dir artifacts/cache/chunks
```
(optional) Submit the job script (e.g., SLURM):

```bash
sbatch run_make_hypergraph.sh
```

While any model can be used, `make_hypergraph.ipynb` supports the approach described in our Methods section by using the Together API to provide hosted LLM inference with `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` for graph generation.


## 3) Run Hypergraph Agents 
Open and run the following notebook in **JupyterLab**:
```bash
Agents.ipynb
```
The agents use: 

Generated hypergraph (compatible with HYPERNETX) can be found in
`artifacts/sg/integrated/final_graph.pkl`

Generated hypergraph with all metadata can be found in
`artifacts/sg/integrated/*_updated_sub_dfs.pkl`

Generated hypergraph embeddings can be found in
`artifacts/sg/graphs/hypergraph_embeedings.pkl`

While any model can be used, `Agents.ipynb` supports the approach described in our Methods section by using the locally hosted LLM model `meta-llama/Llama-3.3-70B-Instruct` for inference. Our embedding model `nomic-ai/nomic-embed-text-v1.5` is also loaded locally using the `SentenceTransformers` library.

## 4) Analyze Hypergraph
Open and run the following notebook in **JupyterLab**:
```bash
analyze_hypergraph.ipynb
```

The analysis uses: 
Generated hypergraph (compatible with HYPERNETX) can be found in
`artifacts/sg/integrated/final_graph.pkl`

The graph and embedding data can also be downloaded as a Hugging Face 🤗 dataset at: [lamm-mit/HyperGraphReasoning](https://huggingface.co/datasets/lamm-mit/HyperGraphReasoning).

***

## Supplementary Output 

Full text output from the agents as seen in the Results section of the manuscript can be found in `Notebooks/SG/Experiments`

## Citation

Please cite this work as:

```bibtex
@article{stewartbuehler2025hypergraphreasoning,
  title     = {Higher-Order Knowledge Representations for Agentic Scientific Reasoning},
  author    = {I.A. Stewart and M.J. Buehler},
  journal   = {arXiv},
  year      = {2026},
  doi       = {https://arxiv.org/abs/2601.04878}
}
