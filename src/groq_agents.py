import json
from typing import Any
from groq import Groq, NotFoundError
from pydantic import BaseModel, field_validator


def _coerce_to_str(v: Any) -> str:
    if isinstance(v, list):
        return "\n".join(str(item) for item in v)
    if isinstance(v, dict):
        return json.dumps(v, indent=2)
    return str(v)


class HypothesisOutput(BaseModel):
    title: str
    rationale: str
    proposed_changes: str

    @field_validator("title", "rationale", "proposed_changes", mode="before")
    @classmethod
    def ensure_string(cls, v: Any) -> str:
        return _coerce_to_str(v)


class CriticOutput(BaseModel):
    approved: bool
    feedback: str

    @field_validator("feedback", mode="before")
    @classmethod
    def ensure_string(cls, v: Any) -> str:
        return _coerce_to_str(v)


class AIScientistPipeline:

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        fallback_models: tuple = ("qwen/qwen3.8-27b", "openai/gpt-oss-20b"),
    ):
        self.client = Groq(api_key=api_key)
        self.models = (model,) + fallback_models

    def _chat(self, system: str, user: str, json_mode: bool = False) -> str:
        last_err = None
        for active_model in self.models:
            try:
                kwargs = {
                    "model": active_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.3,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except NotFoundError as e:
                last_err = e
                continue
        raise RuntimeError(
            f"None of the configured Groq models ({self.models}) are accessible: {last_err}"
        )

    def generate_hypothesis(
        self, baseline_code: str, past_results: str
    ) -> HypothesisOutput:
        sys_prompt = (
            "You are a Principal AI Researcher. Analyze the baseline Python ML script and past results. "
            "Propose ONE concrete, high-impact algorithmic or feature engineering hypothesis to improve val_accuracy. "
            "Respond strictly in JSON matching keys: 'title' (string), 'rationale' (string), 'proposed_changes' (string)."
        )
        raw = self._chat(
            sys_prompt,
            f"Code:\n{baseline_code}\n\nHistory:\n{past_results}",
            json_mode=True,
        )
        return HypothesisOutput(**json.loads(raw))

    def write_code(
        self, baseline_code: str, hypothesis: HypothesisOutput
    ) -> str:
        sys_prompt = (
            "You are an expert ML Engineer using Python 3.11+ and Pandas 2.2+. Modify the baseline script to implement the hypothesis. "
            "Rules: 1) Keep dataset loading and random_state=42 identical (you may add engineered columns after loading df). "
            "2) Must save 'outputs/metrics.json' with 'val_accuracy' and 'outputs/metrics_plot.png'. "
            "3) Only import os (for os.makedirs), json, matplotlib, numpy, pandas, sklearn. "
            "4) NEVER use deprecated Pandas methods like .replace(..., method='bfill') or .fillna(method=...) — use .replace(0, np.nan).bfill() instead. "
            "5) Return ONLY raw valid Python code without markdown fences."
        )
        user_msg = (
            f"Baseline:\n{baseline_code}\n\n"
            f"Hypothesis: {hypothesis.title}\n"
            f"Changes: {hypothesis.proposed_changes}"
        )
        code = self._chat(sys_prompt, user_msg)
        return code.replace("```python", "").replace("```", "").strip()

    def fix_code(self, broken_code: str, error_traceback: str) -> str:
        sys_prompt = (
            "You are a Senior Python ML Debugger. Fix the runtime error in the provided candidate experiment script. "
            "Keep the exact experimental goal, dataset seed=42, and output paths ('outputs/metrics.json' and 'outputs/metrics_plot.png'). "
            "Only import os (for os.makedirs), json, matplotlib, numpy, pandas, sklearn. "
            "NEVER use deprecated Pandas methods like .replace(..., method='bfill') or .fillna(method=...). "
            "Return ONLY the complete, fixed raw Python code without markdown fences."
        )
        user_msg = (
            f"Broken Code:\n{broken_code}\n\n"
            f"Error Traceback:\n{error_traceback}"
        )
        code = self._chat(sys_prompt, user_msg)
        return code.replace("```python", "").replace("```", "").strip()

    def review_results(
        self,
        hypothesis: HypothesisOutput,
        baseline_metrics: dict,
        new_metrics: dict,
    ) -> CriticOutput:
        sys_prompt = (
            "You are a Peer Reviewer. Compare baseline metrics against new experiment metrics. "
            "Approve ONLY if val_accuracy strictly improved without data leakage. "
            "Respond strictly in JSON with keys: 'approved' (boolean), 'feedback' (string)."
        )
        user_msg = (
            f"Hypothesis: {hypothesis.model_dump_json()}\n"
            f"Baseline: {baseline_metrics}\n"
            f"New: {new_metrics}"
        )
        raw = self._chat(sys_prompt, user_msg, json_mode=True)
        return CriticOutput(**json.loads(raw))

    def draft_paper(
        self,
        hypothesis: HypothesisOutput,
        baseline_metrics: dict,
        new_metrics: dict,
        critic: CriticOutput,
        baseline_code: str = "",
    ) -> str:
        sys_prompt = (
            "You are an AI Research Author writing for srinivasta.github.io. "
            "Start the file directly with standard --- YAML frontmatter (title, date, tags) — NEVER wrap frontmatter in ```yaml fences. "
            "Include Abstract, Hypothesis, Methodology, Empirical Results table, and Conclusion. "
            "CRITICAL: Only cite exact hyperparameters, features, and metrics present in the provided code and metrics—never invent dates or values."
        )
        user_msg = (
            f"Baseline Code:\n{baseline_code}\n\n"
            f"Hypothesis: {hypothesis.model_dump_json()}\n"
            f"Baseline Metrics: {baseline_metrics}\n"
            f"New Metrics: {new_metrics}\n"
            f"Reviewer Note: {critic.feedback}"
        )
        return self._chat(sys_prompt, user_msg)
