# CeRAI AIEvaluationTool — Critique Issues

Filed as part of Gates Foundation AI Fellowship – India 2026 Technical Assignment (Option B).
Each issue follows the required format: description, steps to reproduce, impact on evaluation quality, suggested fix.

---

## Issue #1 — Infrastructure Prerequisites Make the Tool Unreproducible in Resource-Constrained Environments

**Labels:** `bug`, `setup`, `reproducibility`

### Description
The tool mandates a stack of infrastructure dependencies that is impractical for independent researchers, fellowship candidates, and low-resource deployments — precisely the contexts in which AI evaluation for development settings is most needed. Required before a single test case can run: MariaDB Server 10.5+, Node.js 20.19+, Google Chrome + ChromeDriver (version-locked to Chrome), Ollama with `qwen3:32b` (≈20 GB), and `sarvam-2b-v0.5` + `shieldgemma-2b` + `sarvam-translate` (additional multi-GB downloads), plus 50 GB free disk and 8 GB RAM minimum (24 GB recommended).

### Steps to Reproduce
1. Clone the repository on a standard Google Colab free-tier environment or a mid-range developer laptop.
2. Follow the README installation steps sequentially.
3. Attempt to `pip install -r requirements.txt` and then start the interface manager.
4. Observe that the process fails or is impossible before any evaluation begins — ChromeDriver version must manually match Chrome, MariaDB requires root setup, and `qwen3:32b` alone exhausts free Colab storage.

### Impact on Evaluation Quality
The tool cannot be evaluated at all in the environments that matter most for the stated use cases (QA teams, compliance officers, AI/ML engineers in resource-limited settings). An evaluation framework that cannot be run cannot produce evaluation results. This is a hard blocker, not a degradation of quality — it is a complete failure to evaluate.

### Suggested Fix
- Make all heavyweight dependencies optional with clearly documented fallback paths.
- SQLite is already supported — make it the unambiguous default with no MariaDB mention needed for a first run.
- Remove the Chrome/ChromeDriver hard dependency for API-only evaluation paths; Selenium is only needed for Web/WhatsApp targets.
- Provide a `requirements-minimal.txt` that installs only what is needed for API evaluation.
- Document a `--dry-run` or `--mock` mode that works without any model running, to let new users verify the pipeline structure.
- Recommend `llama3.2:3b` or `phi3:mini` as the default judge model for resource-constrained environments instead of `qwen3:32b`.

---

## Issue #2 — LLM-as-Judge Is the Only Scoring Mechanism; No Deterministic Fallback Exists

**Labels:** `design`, `reliability`, `evaluation-quality`

### Description
The entire evaluation pipeline depends on an LLM judge (default: `qwen3:32b`) to score responses. There is no mode in which evaluation proceeds without a running model. This creates a circular dependency: you need a large, well-configured LLM to evaluate whether a conversational LLM is performing well. The judge model itself is never validated, calibrated, or checked for consistency. No deterministic or rule-based fallback is provided for any metric, even those (like response latency or keyword presence) that require no LLM at all.

### Steps to Reproduce
1. Configure the tool with a valid target API.
2. Set `LLM_AS_JUDGE_MODEL` in `.env` to an unavailable or incorrect model name.
3. Run `python analyze.py`.
4. Observe that the entire analysis pipeline fails — no partial results are produced for metrics that could be evaluated deterministically (e.g., latency, null-response detection, character count).

### Impact on Evaluation Quality
- **Evaluation is not reproducible**: two runs of the same test suite with different judge model versions will produce different scores with no documented variance.
- **Bias laundering**: the judge LLM has its own biases, yet the tool presents its scores as objective metrics.
- **No ablation**: there is no way to know which scores reflect genuine model quality versus judge model artifacts.
- **Single point of failure**: if the judge model server is down or slow, 100% of evaluation fails.

### Suggested Fix
- Implement a `--no-llm` flag that runs only deterministic metrics (latency, null response, keyword match, length heuristics).
- Add a separate `judge_model_consistency_check` that runs a fixed prompt through the judge and validates the score falls within an expected range.
- Document judge model version alongside every evaluation report so results are reproducible.
- Provide rule-based implementations for at least: toxicity (via keyword list + regex), hallucination detection (entity overlap), and guardrail testing (refusal phrase detection).

---

## Issue #3 — Test Data Is Locked Behind a "Request" Gate; No Public Dataset Ships with the Tool

**Labels:** `documentation`, `usability`, `data`

### Description
The README states: *"A detailed set of Seeding data points shall be provided upon request."* The only shipped data file (`DataPoints.json`) is described as a "sample dataset" with no indication of how many test cases it contains or what domains it covers. An evaluation framework with no evaluation data is not a framework — it is scaffolding. For any external user (researchers, fellows, teams not affiliated with CeRAI IIT Madras), the tool is immediately blocked from meaningful use.

### Steps to Reproduce
1. Clone the repository and inspect `data/DataPoints.json`.
2. Attempt to run a full evaluation using only the shipped data.
3. Observe that the sample data is insufficient to exercise more than a fraction of the 7 test plan categories.
4. Contact the maintainers to "request" the full dataset — note that this makes evaluation non-reproducible by external parties.

### Impact on Evaluation Quality
- Any evaluation results produced with private, unreleased test data cannot be independently verified.
- This violates the basic principle of scientific reproducibility that evaluation benchmarks depend on.
- It creates an asymmetry where CeRAI-internal evaluations are not comparable to any external evaluation.
- A fellowship candidate or external QA team cannot use the tool meaningfully without negotiating access.

### Suggested Fix
- Release at minimum 50 test cases per test plan category (350 total) as a public, versioned benchmark dataset.
- Document the data schema fully so users can generate synthetic test data programmatically.
- Provide a `generate_test_data.py` script that uses an LLM to generate domain-specific test cases from a user-supplied topic and language.
- Tag released datasets with provenance and version numbers so evaluations are traceable.

---

## Issue #4 — WhatsApp and Web Evaluation Paths Depend on Selenium with Hard-Coded XPaths, Making Them Inherently Brittle

**Labels:** `bug`, `automation`, `maintainability`

### Description
The `interface_manager` uses Selenium with XPaths stored in `xpaths.json` to automate WhatsApp Web and arbitrary web applications. The README explicitly instructs users to "Right-click → Inspect → Copy XPath" from their target application. This approach is fundamentally brittle: any UI update to the target application (or to WhatsApp Web, which updates frequently) silently breaks the evaluation pipeline. The tool provides no mechanism to detect that XPaths have become stale, no retry logic documented, and no headless-mode fallback verified across environments.

### Steps to Reproduce
1. Configure a WhatsApp Web evaluation target per README instructions.
2. Note the date and XPaths used.
3. Wait for WhatsApp Web to deploy a UI update (typically every 2-4 weeks).
4. Re-run the same evaluation.
5. Observe that the automation silently fails to locate elements, and the tool either hangs or produces empty response fields with no clear error.

### Impact on Evaluation Quality
- Evaluation results become **silent null** — the tool appears to run but records no responses because element selectors fail without raising exceptions.
- There is no way to distinguish "model gave no response" from "Selenium failed to find the response element" in the output.
- This makes longitudinal evaluations (comparing a model across time) unreliable.
- It also means the WhatsApp evaluation path requires continuous manual maintenance by someone with access to both the UI and the tool config.

### Suggested Fix
- Add an explicit element-detection health check at the start of every Selenium session that verifies all configured XPaths resolve to at least one element before test execution begins.
- Log a `XPATH_STALE` warning with the failing selector whenever an element is not found, rather than silently proceeding.
- For API-type targets, remove Selenium entirely — it is unnecessary.
- Provide a `--validate-selectors` flag that runs a dry-pass through all configured XPaths and reports which ones are broken.
- Consider migrating to Playwright, which has more robust auto-wait semantics and better handling of dynamic DOMs.

---

## Issue #5 — Evaluation Reports Are CLI-Only with No Structured Machine-Readable Output Format

**Labels:** `feature`, `usability`, `interoperability`

### Description
The final step of the pipeline (`report.py`) produces a report that displays in the terminal. No structured output (JSON, CSV, or JSONL) is generated by default. The README mentions an embedded "structured data block" as a goal for strong submissions, but the tool itself does not produce one. This means evaluation results cannot be consumed by downstream systems, compared across runs programmatically, or embedded in CI/CD pipelines without custom parsing of terminal output.

### Steps to Reproduce
1. Complete a full evaluation run.
2. Run `python report.py --config "config.json" --run-name <run-name>`.
3. Observe the output is a formatted terminal display.
4. Attempt to pipe the output to `jq` or parse it programmatically.
5. Observe that there is no `--output-json` or `--output-csv` flag and no documented schema for the output format.

### Impact on Evaluation Quality
- Results cannot be version-controlled or diffed between runs.
- Trend analysis across multiple evaluation runs requires custom scraping of terminal output.
- The tool cannot be integrated into automated testing pipelines (GitHub Actions, Jenkins) without significant wrapper code.
- There is no canonical schema for what an evaluation result *is*, making community adoption difficult.

### Suggested Fix
- Add `--output json` and `--output csv` flags to `report.py`.
- Define and document a canonical JSON schema for an evaluation result object (run_id, timestamp, target, metric_id, score, confidence, judge_model_version).
- Generate a `results_<run_name>.json` file automatically after every analysis run.
- Provide a `compare_runs.py` utility that takes two run names and outputs a diff of metric scores.
- Consider adopting the HELM or ELEUTHER AI evaluation harness output schema for interoperability.
