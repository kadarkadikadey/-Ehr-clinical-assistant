---
title: EHR Clinical Assistant
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
tags:
- openenv
- reinforcement-learning
- medical-ai
---

# EHR Clinical Assistant 🩺
**A High-Fidelity OpenEnv for Safety-Critical Medical Decision Making**

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-v0.2.0-blue)](https://huggingface.co/openenv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The **EHR Clinical Assistant** is a specialized reinforcement learning environment designed to evaluate AI agents on their ability to manage Electronic Health Records (EHR) while navigating complex clinical safety constraints.

---

## 🔬 The Clinical Challenge
In modern healthcare, medical errors are often data errors. This environment simulates the high-stakes responsibility of a clinical assistant:
* **ICD-10 Mapping**: Translating messy, unstructured physician notes into precise billing codes.
* **Safety Auditing**: Identifying when a standard medication is **contraindicated** by a patient's history (e.g., the "Asthma Trap").
* **State Management**: Maintaining an accurate digital twin of the patient's condition across multiple reasoning steps.



---
### See The Visual Verification (The Swagger UI) 

    https://kadarkadikadey-ehr-clinical-assistant.hf.space/docs


---
## 🏗️ Technical Architecture

### State & Action Spaces
The environment follows a partially observable Markov decision process (POMDP) where the agent interacts with a simulated SQL database via a FastAPI bridge.

**Observation Space:**
* `raw_notes`: Unstructured natural language clinical encounter data.
* `diagnoses`: A dynamic list of current ICD-10 strings (e.g., `["I10", "E11"]`).
* `prescriptions`: A dynamic list of medication strings.
* `current_view`: Metadata tracking the episode step count.

**Action Space:**
| Command | Payload | Description |
| :--- | :--- | :--- |
| `ADD_DIAGNOSIS` | `string` | Appends an ICD-10 code to the record. |
| `ADD_MED` | `string` | Appends a medication to the record. |
| `FINISH` | `null` | Submits the record for terminal grading. |

---

## 🎯 Task Registry & Benchmarks

| Task ID | Patient | Complexity | Clinical Objective |
| :--- | :--- | :--- | :--- |
| **`easy_coding`** | P001 | Low | Single-diagnosis extraction (Hypertension). |
| **`medium_triage`** | P002 | Med | Multi-point triage (Diabetes + Metformin). |
| **`hard_reconcile`** | P003 | High | **Safety Audit**: Must identify BP issues without triggering an Asthma-related drug interaction. |

### Baseline Performance (Success Rate)
| Model | Easy | Medium | Hard |
| :--- | :---: | :---: | :---: |
| Random Policy | 10% | 5% | 0% |
| **Qwen-2.5-72B** | **92%** | **78%** | **64%** |
| Perfect Agent | 100% | 100% | 100% |

---

## 🧠 Reward Modeling
This environment utilizes a **Hybrid Dense-Sparse Reward Function**:
1. **Step-wise Dense Reward (0.1)**: Granted for every *unique* and *accurate* clinical finding added.
2. **Anti-Hacking Penalty**: Duplicate actions return a `0.0` reward and a terminal error flag to prevent infinite loops.
3. **Terminal Safety Grader**: Upon calling `FINISH`, the agent is evaluated on a 0.0–1.0 scale. Prescribing a contraindicated medication results in a **Hard Fail** (Score: 0.0).

---

## 🚀 Deployment & Usage

### Local Development
```bash
# Clone the repository
git clone [https://huggingface.co/spaces/YOUR_USER/ehr-clinical-assistant](https://huggingface.co/spaces/YOUR_USER/ehr-clinical-assistant)
cd ehr-clinical-assistant

# Install dependencies via UV (Recommended) or PIP
pip install -r requirements.txt

# Launch the Environment Server
python server/app.py
```
## 🚀 Execution & Evaluation

### Running Inference
To evaluate an AI agent against this environment, use the provided `inference.py` script. This script handles the agent-environment loop, LLM prompting, and real-time logging.

1. **Set Environment Variables**:
   ```bash
   # Your Hugging Face API Token (Required)
   export HF_TOKEN="your_huggingface_token_here"

   # Target Model (Default: Qwen2.5-72B-Instruct)
   export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

   # API Base URL (Hugging Face Inference Endpoint)
   export API_BASE_URL="[https://router.huggingface.co/v1](https://router.huggingface.co/v1)"
   ```

2. **Execute the Inference Script**:
   ```bash
   python inference.py
   ```
3. **Docker**:
   ```bash
   # 1. Build the image
   docker build -t ehr-assistant .

   # 2. Run the container on the standard OpenEnv port
   docker run -p 7860:7860 ehr-assistant
   ```

### Technical Stack
1. **FastAPI** : High-performance asynchronous framework for the OpenEnv API layer.
2. **Uvicorn**: ASGI server implementation for lightning-fast request handling
3. **Pydantic v2**: Strict data validation for clinical schema enforcement
4. **PyTorch**: Powering the RewardPredictor neural network for state-value estimation
5. **UV**: Next-generation Python package installer for 100% reproducible builds and lockfile management.
6. **Docker**: Standardized containerization for multi-mode deployment.
