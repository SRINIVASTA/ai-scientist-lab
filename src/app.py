from datetime import datetime
import os
import streamlit as st

from gdrive_store import GDriveStore
from github_sync import GitHubManager
from groq_agents import AIScientistPipeline
from sandbox_runner import execute_experiment

st.set_page_config(
    page_title="AI Scientist Lab | Srinivasta", page_icon="🔬", layout="wide"
)
st.title("🔬 Autonomous AI Scientist Lab")
st.caption(
    "Automated Hypothesis Generation, Sandboxed Execution & Research Publishing"
)

# Initialize clients from secrets
groq_agent = AIScientistPipeline(api_key=st.secrets["groq"]["api_key"])
gh_mgr = GitHubManager(
    token=st.secrets["github"]["token"],
    repo_name=st.secrets["github"]["repo_name"],
    pages_repo=st.secrets["github"]["pages_repo"],
)
drive = GDriveStore(
    service_account_json_str=st.secrets["gdrive"]["service_account_json"],
    folder_id=st.secrets["gdrive"]["folder_id"],
)

tab_run, tab_drive = st.tabs(["🧪 Experiment Loop", "☁️ Drive Artifacts"])

with tab_run:
    col1, col2 = st.columns([1, 1])

    with open("baselines/baseline_experiment.py", "r") as f:
        baseline_code = f.read()

    with col1:
        st.subheader("Current Baseline Code")
        st.code(baseline_code, language="python")

    with col2:
        st.subheader("Agentic Research Execution")
        if st.button("🚀 Run Autonomous Research Cycle", type="primary"):
            with st.status("Running AI Scientist Loop...", expanded=True) as status:
                st.write("1️⃣ Executing current baseline...")
                base_res = execute_experiment("baselines/baseline_experiment.py")
                base_metrics = base_res.get("metrics", {})
                st.write(f"Baseline Metrics: `{base_metrics}`")

                st.write("2️⃣ Groq Researcher generating hypothesis...")
                hyp = groq_agent.generate_hypothesis(
                    baseline_code, str(base_metrics)
                )
                st.info(f"**Hypothesis:** {hyp.title}\n\n{hyp.rationale}")

                st.write("3️⃣ Groq Engineer writing candidate experiment...")
                candidate_code = groq_agent.write_code(baseline_code, hyp)
                os.makedirs("outputs", exist_ok=True)
                cand_path = "outputs/candidate_experiment.py"
                with open(cand_path, "w") as f:
                    f.write(candidate_code)

                st.write("4️⃣ Running AST safety check & sandboxed execution...")
                cand_res = execute_experiment(cand_path)

                if not cand_res["success"]:
                    st.error(f"Sandbox Execution Failed:\n{cand_res['error']}")
                    status.update(label="Experiment Failed", state="error")
                else:
                    new_metrics = cand_res["metrics"]
                    st.write(f"New Metrics: `{new_metrics}`")

                    st.write("5️⃣ Groq Critic evaluating results...")
                    critic = groq_agent.review_results(
                        hyp, base_metrics, new_metrics
                    )

                    if critic.approved:
                        st.success(f"Approved by Critic: {critic.feedback}")
                        paper_md = groq_agent.draft_paper(
                            hyp, base_metrics, new_metrics, critic
                        )

                        # Upload artifacts to Google Drive
                        plot_link = drive.upload_file(
                            "outputs/metrics_plot.png", "image/png"
                        )
                        metrics_link = drive.upload_file(
                            "outputs/metrics.json", "application/json"
                        )

                        # Create GitHub PR
                        branch = f"exp-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
                        pr_body = (
                            f"### Hypothesis\n{hyp.rationale}\n\n"
                            f"### Metrics\n- Baseline: `{base_metrics}`\n- Candidate: `{new_metrics}`\n\n"
                            f"### Drive Artifacts\n- [Plot]({plot_link}) | [Metrics JSON]({metrics_link})"
                        )
                        pr_url = gh_mgr.create_experiment_pr(
                            branch, candidate_code, paper_md, hyp.title, pr_body
                        )

                        st.session_state["latest_paper"] = paper_md
                        st.session_state["latest_slug"] = branch
                        st.markdown(f"✅ **Pull Request Opened:** [{pr_url}]({pr_url})")
                        status.update(
                            label="Research Cycle Complete!", state="complete"
                        )
                    else:
                        st.warning(f"Rejected by Critic: {critic.feedback}")
                        status.update(
                            label="Hypothesis Rejected", state="complete"
                        )

        if "latest_paper" in st.session_state:
            st.divider()
            st.subheader("📄 Generated Research Paper")
            st.markdown(st.session_state["latest_paper"])
            if st.button("🌐 Publish Report to srinivasta.github.io"):
                live_url = gh_mgr.publish_to_portfolio(
                    st.session_state["latest_slug"],
                    st.session_state["latest_paper"],
                )
                st.success(f"Published live to portfolio: {live_url}")

with tab_drive:
    st.subheader("Google Drive Experiment Archive")
    if st.button("🔄 Refresh Artifacts"):
        files = drive.list_artifacts()
        for f in files:
            st.markdown(
                f"- **{f['name']}** ({f['createdTime']}) — [View in Drive]({f['webViewLink']})"
            )
