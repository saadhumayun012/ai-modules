# AI Backend for Academic Document Analysis

A pure REST API (no UI) that processes .docx files through three modules: grammar checking, coherence analysis, and a RAG-powered chatbot. This repository is part of my final year project

## Stack
FastAPI · spaCy · T5-base (Modal.com GPU) · FastEmbed · Qdrant · Groq · Docker

## Project Structure
```
app/
├── core/        settings, constants, models, qdrant client, embeddings
├── routes/      grammar, coherence, chat indexing, chat query
├── services/    document extractor, grammar, coherence, indexing
└── utils/       sentence splitter, file validator
```

---

## Module 1 – Grammar Checker
`POST /grammar/check-document`

Upload a `.docx`; get back corrected sentences with heading location and word‑level diffs.

The T5 model runs on Modal.com (GPU). All preprocessing, filtering, and diff generation run locally.

**Request:**
```
Content-Type: multipart/form-data
file: thesis.docx
```

**Response:**
```json
{
  "filename": "thesis.docx",
  "total_sections": 12,
  "total_corrections": 3,
  "corrections": [
    {
      "heading": "Research Methodology",
      "original": "The data was collected using survey.",
      "corrected": "The data was collected using a survey.",
      "highlighted": "The data was collected using [ADDED -> a] survey.",
      "changes": [{ "type": "insert", "original": "", "corrected": "a" }],
      "confidence": 0.94
    },
    {
      "heading": "1.2 Scope",
      "original": "The findings suggests a strong correlation.",
      "corrected": "The findings suggest a strong correlation.",
      "highlighted": "The findings [suggests -> suggest] a strong correlation.",
      "changes": [{ "type": "replace", "original": "suggests", "corrected": "suggest" }],
      "confidence": 0.97
    }
  ]
}
```

**Skipped content:** headers, TOC, figure/table captions, references, bullet lists, technical labels (`UC-1:`, `REQ-1:`), URLs, fragments without a verb, and short table cells.

**Protected tokens:** citations (`Smith (2020)`) and file extensions (`.docx`) are masked before inference and restored after, so the model does not alter them.

**Model details:** T5‑base fine‑tuned on JFLEG + CoEdIT (~21K pairs), trained for 5 epochs on Colab T4 (best checkpoint at epoch 3). Deployed as a serverless GPU function on Modal.com. Cold start adds ~15s after idle.

---

## Module 2 – Coherence Analyzer
`POST /coherence`

Upload a `.docx`; get back sentence‑ and paragraph‑level coherence issues detected using embedding similarity.

Tables are excluded because their rows deliberately cover different topics; comparing them would produce false positives.

**Request:**
```
Content-Type: multipart/form-data
file: thesis.docx
```

**Response:**
```json
{
  "filename": "thesis.docx",
  "total_issues": 2,
  "issues": [
    {
      "heading": "Literature Review",
      "level": "sentence",
      "location": "Paragraph 2, Sentence 1 -> 3",
      "score": 0.43,
      "sentence_1": "Early studies focused on rule-based grammar systems.",
      "sentence_2": "The database is normalized to third normal form."
    },
    {
      "heading": "Methodology",
      "level": "paragraph",
      "location": "Paragraph 3 -> 4",
      "score": 0.38
    }
  ]
}
```

Scores below 0.5 (sentence) or 0.4 (paragraph) are flagged. Both thresholds are configurable via settings.py.

---

## Module 3 – RAG Chatbot
Two endpoints: index once, query many times.

### `POST /chat/index`
```json
{
  "filename": "thesis.docx",
  "total_chunks": 142,
  "message": "Document indexed successfully."
}
```

### `POST /chat/query`
**Request:**
```json
{ "question": "What methodology was used?", "document_id": "thesis" }
```

**Response:**
```json
{
  "question": "What methodology was used?",
  "answer": "The study used a mixed-methods approach combining structured interviews and document analysis...",
  "sources": [
    { "heading": "3.1 Research Design", "chunk": "The study used a mixed-methods approach..." }
  ]
}
```

**Retrieval pipeline:** query expansion → vector search per variation → deduplicate overlapping chunks → rerank by keyword + semantic score → Groq generates the final answer.

---

## Shared DOCX Parser
All three modules use the same `extract_structure()` function, which walks the document body in order and groups content under headings.

**Heading detection (priority order):**
1. Word heading styles (`Heading 1` through `Heading 6`, `Title`, etc.)
2. Fallback: bold + short line, large font, ALL CAPS, numbered pattern (e.g. `1.1 Title`)

**Always skipped:** TOC, captions, references, and anything listed in `SKIP_STYLES` / `SKIP_HEADINGS` in `constants.py`.

**Tables:** included for grammar and chatbot, excluded for coherence.

---

`.env` **file**:
```bash
BASE_URL=https://api.groq.com/openai/v1
API_KEY=api-key
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=rag_documents
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=llama-3.1-8b-instant
GRAMMAR_API_URL=where-model-is-deploy
```

### Model
The fine‑tuned grammar model is available at:  
[https://huggingface.co/sahdee/fyp-t5-grammar-base](https://huggingface.co/sahdee/fyp-t5-grammar-base)

---

## Modal Deployment Code
The grammar model is deployed as a serverless GPU function on Modal. See the script below:

```python
import modal

app = modal.App("fyp-grammar-inference")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.2.1",
        "transformers==4.38.2",
        "sentencepiece==0.2.0",
        "protobuf==3.20.3",
        "numpy<2",
        "fastapi",
        "pydantic",
    )
)

MODEL_VOLUME = modal.Volume.from_name("grammar-model-vol")

@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
    scaledown_window=60,
    volumes={"/model": MODEL_VOLUME},
)
class GrammarModel:

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        print("Loading model...")

        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.tokenizer.padding_side = "left"

        self.model = T5ForConditionalGeneration.from_pretrained("/model/model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval().half()
        print(f" ==>Model loaded on {self.device}")

    @modal.fastapi_endpoint(method="GET", label="health")
    def health(self):
        return {"status": "ready", "device": str(self.device)}

    @modal.fastapi_endpoint(method="POST", label="predict-batch")
    def predict_batch(self, data: dict):
        import torch
        texts = data.get("texts", [])

        if not texts:
            return {"predictions": []}

        SUB_BATCH_SIZE = 32
        all_predictions = []

        for i in range(0, len(texts), SUB_BATCH_SIZE):
            sub_batch = texts[i : i + SUB_BATCH_SIZE]

            inputs = self.tokenizer(
                [f"grammar: {t}" for t in sub_batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            results = self.tokenizer.batch_decode(
                outputs.sequences,
                skip_special_tokens=True
            )
            confidences = (
                torch.exp(outputs.sequences_scores).tolist()
                if outputs.sequences_scores is not None
                else [1.0] * len(results)
            )

            for res, conf in zip(results, confidences):
                all_predictions.append({"corrected": res, "confidence": round(conf, 4)})

            del inputs, outputs
            torch.cuda.empty_cache()

        return {"predictions": all_predictions}


@app.local_entrypoint()
def main():
    print("Deploy with: modal deploy modal_app.py")
```

---

## Limitations
- Only `.docx` files are supported – no PDF or plain text.
- Documents without proper Word heading styles will still work, but heading names in responses will be less precise.
- The chatbot only knows content from the indexed document; it cannot answer general questions.

---
