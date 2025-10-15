# Self‑Healing Classification DAG

A robust, self‑healing text classification pipeline that pairs a fine‑tuned DeBERTa‑v3 model with a LangGraph workflow. The system is designed for human‑in‑the‑loop (HIL) scenarios where correctness matters more than blind automation: if the model is uncertain, the pipeline asks a human for clarification and records the result.

---

## Fine‑Tuned Model: [ragunath‑ravi/deberta‑v3‑emotion‑classifier](https://huggingface.co/ragunath-ravi/deberta-v3-emotion-classifier)

## 🚀 Highlights

* **Fine‑Tuned Model** — Uses a `deberta-v3-base` model fine‑tuned on the `dair-ai/emotion` dataset (hosted on Hugging Face).
* **LangGraph Workflow** — The classification logic is modeled as a Directed Acyclic Graph (DAG) for predictable, stateful flow control.
* **Confidence‑Based Fallback** — If model confidence is below 80%, the pipeline triggers a human fallback loop to “heal” the prediction.
* **Interactive CLI** — Lightweight command‑line UI for quick human‑in‑the‑loop corrections.
* **Structured Logging** — All predictions, fallbacks, and final labels are logged to `app.log` for traceability.

---

## Architecture Overview

The system is implemented as a LangGraph state machine composed of three key nodes:

1. **Inference Node**

   * Receives raw text input.
   * Runs the fine‑tuned DeBERTa model and computes a confidence score.
2. **Confidence Check (Conditional Edge)**

   * Routes execution based on confidence:

     * `>= 80%` → accept the model prediction and finish.
     * `< 80%` → route to the Fallback Node.
3. **Fallback Node**

   * Presents the uncertain prediction to a human user via the CLI.
   * Accepts a corrected label and returns it as the final output.

This design keeps the automated flow as the default while providing a clear, auditable path when human judgment is required.

---

## Installation & Setup

**Prerequisites**

* Python 3.8+
* `pip`
* `venv` (optional but recommended)

**Clone the repo**

```bash
git clone https://github.com/your-username/self-healing-dag.git
cd self-healing-dag
```

**Create and activate a virtual environment**

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell / CMD):

```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

Start the interactive CLI from the project root:

```bash
python main.py
```

On first run the script will download and cache the fine‑tuned model from Hugging Face.

### Example: High Confidence (no HIL)

```
Input: I am so happy and excited for the event!
[InferenceNode] Predicted label: joy | Confidence: 99%
Final Label: joy (High confidence)
```

### Example: Low Confidence (fallback engaged)

```
Input: I’m so done dealing with this nonsense every single day
[InferenceNode] Predicted label: sadness | Confidence: 77%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? The model predicted 'sadness': anger
User: anger
Final Label: anger (Corrected via user clarification)
```

---

## Development Notes & Tips

* The confidence threshold is configurable — raising it increases human workload but improves correctness; lowering it reduces HIL involvement.
* Keep `app.log` under version control only if you intentionally want to share run traces; otherwise add it to `.gitignore`.
* For production deployments consider adding:

  * An authentication layer for human annotators
  * A web UI that surfaces uncertain predictions and annotator history
  * Batch review mode for low‑confidence items

---

## License


---

