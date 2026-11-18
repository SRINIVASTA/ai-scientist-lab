from datetime import datetime
import difflib
import os
import streamlit as st

from gdrive_store import GDriveStore
from github_sync import GitHubManager
from groq_agents import AIScientistPipeline, CriticOutput, HypothesisOutput
from sandbox_runner import execute_experiment

st.set_page_config(
    page_title="AI Scientist Lab | Srinivasta",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Autonomous AI Scientist Lab")
st.caption(
    "Automated Hypothesis Generation, Sandboxed Execution & Research Publishing"
)


def is_dummy(val: str) -> bool:
    v = str(val).strip()
    return (
        not v
        or "your_" in v
        or "PASTE_YOUR" in v
        or "..." in v
        or v in {"gsk_your_groq_api_key_here", "ghp_your_github_pat_here"}
    )


def generate_unified_diff(old_code: str, new_code: str) -> str:
    diff = difflib.unified_diff(
        old_code.splitlines(),
        new_code.splitlines(),
        fromfile="baseline_experiment.py",
        tofile="candidate_experiment.py",
        lineterm="",
    )
    return "\n".join(diff) or "# No code changes detected"


# 1. Detect Live vs. Dummy Secrets
groq_key = st.secrets.get("groq", {}).get("api_key", "")
gh_token = st.secrets.get("github", {}).get("token", "")
gh_repo = st.secrets.get("github", {}).get(
    "repo_name", "srinivasta/ai-scientist-lab"
)
gh_pages = st.secrets.get("github", {}).get(
    "pages_repo", "srinivasta/srinivasta.github.io"
)

groq_live = not is_dummy(groq_key)
gh_live = not is_dummy(gh_token)

groq_agent = AIScientistPipeline(api_key=groq_key.strip()) if groq_live else None
gh_mgr = (
    GitHubManager(
        token=gh_token.strip(),
        repo_name=gh_repo.strip(),
        pages_repo=gh_pages.strip(),
    )
    if gh_live
    else None
)

drive = None
if "gdrive" in st.secrets and not is_dummy(
    st.secrets["gdrive"].get("folder_id", "")
):
    sa_creds = (
        st.secrets["gdrive"]["service_account"]
        if "service_account" in st.secrets["gdrive"]
        else st.secrets["gdrive"].get("service_account_json")
    )
    if sa_creds and not is_dummy(str(sa_creds)):
        try:
            drive = GDriveStore(
                service_account_info=sa_creds,
                folder_id=st.secrets["gdrive"]["folder_id"].strip(),
            )
        except Exception:
            drive = None

with st.sidebar:
    st.header("🔌 Service Connections")
    st.markdown(
        f"- **Groq LLM:** {'🟢 Live' if groq_live else '🟡 Demo Mode (Dummy Key)'}"
    )
    st.markdown(
        f"- **GitHub Sync:** {'🟢 Live' if gh_live else '🟡 Demo Mode (Dummy Key)'}"
    )
    st.markdown(
        f"- **Google Drive:** {'🟢 Live' if drive else '🟡 Disabled (Dummy Key)'}"
    )
    st.info(
        "Replace dummy values in **Manage app -> Settings -> Secrets** whenever you are ready to go live."
    )
# 2. Main UI Tabs
tab_run, tab_drive = st.tabs(["🧪 Experiment Loop", "☁️ Drive Artifacts"])

with tab_run:
    col1, col2 = st.columns(2)

    baseline_path = "baselines/baseline_experiment.py"
    if not os.path.exists(baseline_path):
        st.error(f"Baseline file not found at `{baseline_path}`.")
        st.stop()

    with open(baseline_path, "r") as f:
        baseline_code = f.read()

    with col1:
        st.subheader("Current Baseline Code")
        st.code(baseline_code, language="python")

    with col2:
        st.subheader("Agentic Research Execution")
        if st.button("🚀 Run Autonomous Research Cycle", type="primary"):
            with st.status(
                "Running AI Scientist Loop...", expanded=True
            ) as status:
                st.write("1️⃣ Executing current baseline in AST sandbox...")
                base_res = execute_experiment(baseline_path)
                if not base_res["success"]:
                    st.error(
                        f"Baseline Execution Failed:\n{base_res['error']}"
                    )
                    status.update(label="Baseline Failed", state="error")
                    st.stop()

                base_metrics = base_res.get("metrics", {})
                st.write(f"Baseline Metrics: `{base_metrics}`")

                if groq_live:
                    st.write("2️⃣ Groq Researcher generating hypothesis...")
                    hyp = groq_agent.generate_hypothesis(
                        baseline_code, str(base_metrics)
                    )
                    st.info(f"**Hypothesis:** {hyp.title}\n\n{hyp.rationale}")

                    st.write("3️⃣ Groq Engineer writing candidate experiment...")
                    candidate_code = groq_agent.write_code(baseline_code, hyp)
                else:
                    st.write("2️⃣ [Demo Mode] Generating sample hypothesis...")
                    hyp = HypothesisOutput(
                        title="Gradient Boosting with Regime Volatility Features",
                        rationale="Replacing shallow RandomForest with GradientBoostingClassifier better captures nonlinear return-volatility interactions.",
                        proposed_changes="Use GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=3, random_state=42).",
                    )
                    st.info(f"**Hypothesis:** {hyp.title}\n\n{hyp.rationale}")

                    st.write("3️⃣ [Demo Mode] Preparing candidate script...")
                    candidate_code = baseline_code.replace(
                        "from sklearn.ensemble import RandomForestClassifier",
                        "from sklearn.ensemble import GradientBoostingClassifier",
                    ).replace(
                        "RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)",
                        "GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=3, random_state=42)",
                    )

                os.makedirs("outputs", exist_ok=True)
                cand_path = "outputs/candidate_experiment.py"
                with open(cand_path, "w") as f:
                    f.write(candidate_code)

                st.write(
                    "4️⃣ Running AST safety check & sandboxed execution..."
                )
                cand_res = execute_experiment(cand_path)

                if not cand_res["success"]:
                    st.error(
                        f"Sandbox Execution Failed:\n{cand_res['error']}"
                    )
                    status.update(label="Experiment Failed", state="error")
                else:
                    new_metrics = cand_res["metrics"]
                    st.write(f"New Metrics: `{new_metrics}`")

                    if groq_live:
                        st.write("5️⃣ Groq Critic evaluating results...")
                        critic = groq_agent.review_results(
                            hyp, base_metrics, new_metrics
                        )
                    else:
                        improved = new_metrics.get(
                            "val_accuracy", 0
                        ) >= base_metrics.get("val_accuracy", 0)
                        critic = CriticOutput(
                            approved=improved,
                            feedback="[Demo Mode] Out-of-sample directional accuracy held/improved with strict chronological split.",
                        )

                    st.session_state["latest_diff"] = generate_unified_diff(
                        baseline_code, candidate_code
                    )
                    st.session_state["base_metrics"] = base_metrics
                    st.session_state["new_metrics"] = new_metrics

                    if critic.approved:
                        st.success(f"Approved by Critic: {critic.feedback}")
                        if groq_live:
                            paper_md = groq_agent.draft_paper(
                                hyp, base_metrics, new_metrics, critic
                            )
                        else:
                            paper_md = (
                                f"---\ntitle: \"{hyp.title}\"\ndate: {datetime.utcnow().strftime('%Y-%m-%d')}\ntags: [ai-scientist, quant-ml]\n---\n\n"
                                f"## Abstract\nAutonomous experiment testing **{hyp.title}**.\n\n"
                                f"## Empirical Results\n- **Baseline:** `{base_metrics}`\n- **Candidate:** `{new_metrics}`\n\n"
                                f"## Conclusion\n{critic.feedback}"
                            )

                        plot_link, metrics_link = "Demo-Local", "Demo-Local"
                        if drive:
                            plot_link = drive.upload_file(
                                "outputs/metrics_plot.png", "image/png"
                            )
                            metrics_link = drive.upload_file(
                                "outputs/metrics.json", "application/json"
                            )

                        branch = (
                            f"exp-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
                        )
                        if gh_live:
                            pr_body = (
                                f"### Hypothesis\n{hyp.rationale}\n\n"
                                f"### Metrics\n- Baseline: `{base_metrics}`\n- Candidate: `{new_metrics}`\n\n"
                                f"### Drive Artifacts\n- [Plot]({plot_link}) | [Metrics JSON]({metrics_link})"
                            )
                            pr_url = gh_mgr.create_experiment_pr(
                                branch,
                                candidate_code,
                                paper_md,
                                hyp.title,
                                pr_body,
                            )
                            st.markdown(
                                f"✅ **Pull Request Opened:** [{pr_url}]({pr_url})"
                            )
                        else:
                            st.info(
                                f"🟡 **[Demo Mode]** Skipped GitHub PR creation for branch `{branch}`."
                            )

                        st.session_state["latest_paper"] = paper_md
                        st.session_state["latest_slug"] = branch
                        status.update(
                            label="Research Cycle Complete!", state="complete"
                        )
                    else:
                        st.warning(f"Rejected by Critic: {critic.feedback}")
                        status.update(
                            label="Hypothesis Rejected", state="complete"
                        )

    # Visual Diff & Diagnostic Plot Section
    if "latest_diff" in st.session_state:
        st.divider()
        vis_col1, vis_col2 = st.columns([3, 2])
        with vis_col1:
            st.subheader("🔍 Candidate Code Diff")
            st.code(st.session_state["latest_diff"], language="diff")
        with vis_col2:
            st.subheader("📊 Validation Metrics Plot")
            if os.path.exists("outputs/metrics_plot.png"):
                st.image(
                    "outputs/metrics_plot.png",
                    caption=f"Baseline: {st.session_state['base_metrics']} -> Candidate: {st.session_state['new_metrics']}",
                    use_container_width=True,
                )

    if "latest_paper" in st.session_state:
        st.divider()
        st.subheader("📄 Generated Research Paper")
        st.markdown(st.session_state["latest_paper"])
        if st.button("🌐 Publish Report to srinivasta.github.io"):
            if gh_live:
                live_url = gh_mgr.publish_to_portfolio(
                    st.session_state["latest_slug"],
                    st.session_state["latest_paper"],
                )
                st.success(f"Published live to portfolio: {live_url}")
            else:
                st.warning(
                    "🟡 **[Demo Mode]** Add a real GitHub PAT in Secrets to publish directly to `srinivasta.github.io`."
                )

with tab_drive:
    st.subheader("Google Drive Experiment Archive")
    if not drive:
        st.info(
            "Google Drive is currently in **Demo Mode** because dummy credentials are configured. "
            "Add your real GCP Service Account credentials in Streamlit Secrets to view and store Drive artifacts."
        )
    elif st.button("🔄 Refresh Artifacts"):
        try:
            files = drive.list_artifacts()
            if not files:
                st.info(
                    "No artifacts found in the configured Drive folder yet."
                )
            for f in files:
                st.markdown(
                    f"- **{f['name']}** ({f['createdTime']}) — [View in Drive]({f['webViewLink']})"
                )
        except Exception as e:
            st.error(f"Could not fetch Drive artifacts: {e}")
