import json
from groq import Groq
from pydantic import BaseModel


class HypothesisOutput(BaseModel):
    title: str
    rationale: str
    proposed_changes: str


class CriticOutput(BaseModel):
    approved: bool
    feedback: str


class AIScientistPipeline:

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model

    def _chat(self, system: str, user: str, json_mode: bool = False) -> str:
        kwargs = {
            "model": self.model,
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

    def generate_hypothesis(
        self, baseline_code: str, past_results: str
    ) -> HypothesisOutput:
        sys_prompt = (
            "You are a Principal AI Researcher. Analyze the baseline Python ML script and past results. "
            "Propose ONE concrete, high-impact algorithmic or feature engineering hypothesis to improve val_accuracy. "
            "Respond strictly in JSON matching keys: 'title', 'rationale', 'proposed_changes'."
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
            "You are an expert ML Engineer. Modify the baseline script to implement the hypothesis. "
            "Rules: 1) Keep dataset generation and random_state=42 identical. "
            "2) Must save 'outputs/metrics.json' with 'val_accuracy' and 'outputs/metrics_plot.png'. "
            "3) Only import os (for os.makedirs), json, matplotlib, numpy, pandas, sklearn. "
            "4) Return ONLY raw valid Python code without markdown fences."
        )
        user_msg = f"Baseline:\n{baseline_code}\n\nHypothesis: {hypothesis.title}\nChanges: {hypothesis.proposed_changes}"
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
        user_msg = f"Hypothesis: {hypothesis.model_dump_json()}\nBaseline: {baseline_metrics}\nNew: {new_metrics}"
        raw = self._chat(sys_prompt, user_msg, json_mode=True)
        return CriticOutput(**json.loads(raw))

    def draft_paper(
        self,
        hypothesis: HypothesisOutput,
        baseline_metrics: dict,
        new_metrics: dict,
        critic: CriticOutput,
    ) -> str:
        sys_prompt = (
            "You are an AI Research Author writing for srinivasta.github.io. "
            "Write a crisp, publication-grade Markdown research note with YAML frontmatter "
            "(title, date, tags). Include Abstract, Hypothesis, Methodology, Empirical Results table, and Conclusion."
        )
        user_msg = (
            f"Hypothesis: {hypothesis.model_dump_json()}\n"
            f"Baseline: {baseline_metrics}\nNew Metrics: {new_metrics}\nReviewer Note: {critic.feedback}"
        )
        return self._chat(sys_prompt, user_msg)
