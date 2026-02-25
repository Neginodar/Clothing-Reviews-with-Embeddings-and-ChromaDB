# Topic Analysis of Clothing Reviews with Embeddings and ChromaDB

Minimal but advanced **notebook showcase** of using OpenAI embeddings and **ChromaDB** to analyze e‑commerce clothing reviews.

The notebook walks through:

- Creating text embeddings for thousands of customer reviews using OpenAI’s `text-embedding-3-small`.
- Performing topic analysis (Quality / Fit / Style / Comfort) directly in embedding space.
- Visualizing reviews with **t‑SNE** by topic.
- Storing reviews in **ChromaDB** with rich metadata (rating, age, department, topic).
- Running semantic search with metadata filters and multi‑collection design.
- Implementing a simple **RAG‑style QA** over reviews with a lightweight hallucination guard.

---

## 1. Project overview

**Goal**  
Explore how far you can go with “just” embeddings + a vector database + a small LLM call, without full LangChain or a backend service.

**What this notebook demonstrates**

- End‑to‑end flow: dataset → embeddings → topics → visualization → vector DB → search → RAG answer.
- Use of **ChromaDB** as a semantic search layer with metadata, multiple collections, and filtered queries.
- Simple but explicit **RAG patterns**: retrieve → build context → call chat model → apply a small hallucination heuristic.

This is intended as a **focused demo** rather than a full production system.

---

## 2. Data

The project uses the public **Women’s Clothing E‑Commerce Reviews** dataset (Kaggle/DataCamp variant), restricted to a few columns:

- `Review Text` – free‑text customer review  
- `Rating` – integer rating (1–5)  
- `Age` – reviewer age  
- `Department Name` – product department (e.g. Dresses, Tops)  
- `Class Name` – product class (e.g. Blouses, Pants)

The notebook expects a file named:

```text
womens_clothing_e-commerce_reviews.csv
```

in the project root.

---

## 3. Techniques and architecture

### 3.1 Embeddings

- Model: `text-embedding-3-small` (OpenAI) for review and topic embeddings.
- Embeddings are requested in **batches** for efficiency.
- Topic labels (`["Quality", "Fit", "Style", "Comfort"]`) are also embedded and used for nearest‑neighbor topic assignment.

### 3.2 Topic assignment

For each review embedding:

- Compute cosine distance to each topic embedding.
- Assign the topic with **minimum** cosine distance.
- Store the result as a `Topic` column in the DataFrame.

This gives a simple, fully embedding‑based topic classifier without supervised training.

### 3.3 Visualization (t‑SNE)

- Reduce embeddings to 2D using **t‑SNE** from scikit‑learn.
- Color points by `Topic`.
- Provide a second visualization where **nearest neighbors** of a query review are highlighted on the plot.

This illustrates how semantically related reviews cluster in the embedding space.

### 3.4 Vector store (ChromaDB)

Chroma is used as the vector database.

The notebook creates:

1. A main collection **`review_embeddings`** with:
   - `documents`: raw review texts  
   - `metadatas`: rating, age, topic, department, class_name  
   - `ids`: row indices as strings  
   - `embedding_function`: `OpenAIEmbeddingFunction` so Chroma handles embedding internally

2. Two additional collections:
   - **`reviews_positive`** – reviews with rating ≥ 4  
   - **`reviews_negative`** – reviews with rating < 4  

This shows multi‑collection design for fast “only positive” / “only negative” retrieval.

### 3.5 Semantic search and analytics

The notebook defines:

- `find_similar_reviews(input_text, coll, n, where=None)`  
  - Performs semantic search over any collection.  
  - Optional `where` filter on metadata (e.g. `{"rating": {"$gte": 4}, "topic": "Comfort"}`).

- `top_departments_for_topic(topic, coll, min_rating, n)`  
  - Uses Chroma query + metadata to compute simple aggregates (e.g. top departments for “Comfort” in high‑rating reviews).

There is also a small **session** abstraction:

- `ReviewSearchSession` keeps a rolling history of queries and their results, mimicking a tiny search session layer.

### 3.6 RAG‑style question answering

The advanced function:

```python
ask_reviews_question_advanced(
    question: str,
    coll,
    k: int = 10,
    topic: str | None = None,
    sentiment: str | None = None,  # "positive" / "negative" / None
)
```

implements a minimal RAG pipeline:

1. **Collection & filter selection**  
   - Chooses `collection`, `pos_collection`, or `neg_collection` depending on `sentiment`.  
   - Builds `where` filters if `topic` is provided.

2. **Retrieval**  
   - Calls Chroma `.query` with `query_texts=[question]` and `n_results=k`.  
   - Formats retrieved reviews with metadata as context.

3. **Generation**  
   - Calls `gpt-4.1-mini` (configurable) with a system prompt that forbids inventing facts.  
   - Asks the model to answer *only* from the provided reviews.

4. **Hallucination guard (simple heuristic)**  
   - Tokenizes context and answer.  
   - Computes overlap ratio between answer tokens and context tokens.  
   - Sets a `hallucination_flag` when overlap < 0.25 (threshold tunable).

Return structure:

```python
{
    "answer": str,
    "supporting_reviews": List[{"text": str, "meta": dict}],
    "hallucination_flag": bool,
    "overlap_ratio": float,
}
```

---

## 4. Repository structure

Suggested structure:

```text
.
├─ chroma_clothing_showcase.ipynb    # main notebook
├─ womens_clothing_e-commerce_reviews.csv
├─ images/
│  └─ clothing.jpg                   # used in the intro markdown
├─ chroma_reviews_db/                # Chroma persistent data (created at runtime, gitignored)
├─ README.md
└─ .gitignore
```

Example `.gitignore` entries:

```gitignore
chroma_reviews_db/
__pycache__/
.ipynb_checkpoints/
.env
```

---

## 5. How to run

### 5.1 Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 5.2 Create and activate a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scriptsctivate
```

### 5.3 Install dependencies

This notebook uses:

- `pandas`, `numpy`  
- `matplotlib`, `scikit-learn`  
- `chromadb`  
- `openai` (new SDK)

You can install them with:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
pandas
numpy
matplotlib
scikit-learn
chromadb==0.4.17
pysqlite3-binary==0.5.2
openai>=1.0.0
```

### 5.4 Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."   # macOS/Linux
# or
set OPENAI_API_KEY=sk-...       # Windows cmd
```

### 5.5 Run the notebook

```bash
jupyter lab
# or
jupyter notebook
```

Open `chroma_clothing_showcase.ipynb` and run all cells from top to bottom.

---

## 6. Example usage

Inside the notebook, you can try:

```python
# Basic similarity
example_review = "Absolutely wonderful - silky and sexy and comfortable"
similar = find_similar_reviews(example_review, collection, n=3)

# Topic/question-specific RAG
res_negative_fit = ask_reviews_question_advanced(
    "What issues do customers report with dress fit?",
    collection,
    k=15,
    topic="Fit",
    sentiment="negative",
)

print("Answer:
", res_negative_fit["answer"])
print("Hallucination flag:", res_negative_fit["hallucination_flag"])
print("Overlap ratio:", res_negative_fit["overlap_ratio"])
```

You can also inspect the `supporting_reviews` field to see which reviews were used as context.

---

## 7. Limitations and possible extensions

**Limitations**

- Notebook‑only; no API, tests, or deployment.  
- Topic assignment is unsupervised and uses only a few coarse labels.  
- Hallucination guard is purely lexical and heuristic.

**Possible extensions**

- Wrap the core retrieval & QA logic in a **FastAPI** service.  
- Add a tiny **evaluation harness** (fixed questions + human‑labeled answers).  
- Swap OpenAI embeddings with a local model (e.g. via sentence‑transformers) for an API‑free version.  
- Turn topic assignment into a supervised or semi‑supervised classifier.

---

## 8. References

- OpenAI embeddings docs and use cases.
- OpenAI + Chroma vector search examples.
- ChromaDB embedding functions and metadata filtering docs.
- General ChromaDB tutorials and guides.
