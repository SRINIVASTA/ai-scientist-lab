from datetime import datetime
import difflib
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import yfinance as yf

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

if "experiment_history" not in st.session_state:
    st.session_state["experiment_history"] = []


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


def compute_live_forecast(
    csv_path: str = "data/market_data.csv", horizon_days: int = 5
):
    """Trains the multi-scale feature model on all available history and forecasts N trading days ahead."""
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if len(df) < 60 or "Close" not in df.columns:
        return None

    df["ret_1d"] = df["Close"].pct_change(1)
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_20d"] = df["Close"].pct_change(20)
    df["vol_5d"] = df["ret_1d"].rolling(5).std()
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["sma_ratio"] = df["Close"] / df["Close"].rolling(20).mean() - 1.0

    for lag in (1, 2, 3):
        df[f"ret_1d_lag{lag}"] = df["ret_1d"].shift(lag)
        df[f"ret_5d_lag{lag}"] = df["ret_5d"].shift(lag)

    feature_cols = [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "vol_5d",
        "vol_10d",
        "vol_20d",
        "sma_ratio",
        "ret_1d_lag1",
        "ret_1d_lag2",
        "ret_1d_lag3",
        "ret_5d_lag1",
        "ret_5d_lag2",
        "ret_5d_lag3",
    ]

    latest_row = df[feature_cols].dropna().iloc[[-1]]
    latest_close = float(df["Close"].iloc[-1])
    latest_date = str(df["Date"].iloc[-1]) if "Date" in df.columns else "Latest"
    daily_vol = float(df["vol_10d"].dropna().iloc[-1])

    df["target"] = (df["Close"].shift(-horizon_days) > df["Close"]).astype(int)
    train_df = df.dropna(subset=feature_cols + ["target"])

    model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(train_df[feature_cols], train_df["target"])

    probs = model.predict_proba(latest_row).ravel().tolist()
    prob_up = float(probs.pop())
    direction = "BULLISH (UP)" if prob_up >= 0.5 else "BEARISH (DOWN)"
    confidence = prob_up if prob_up >= 0.5 else (1.0 - prob_up)

    sigma_horizon = daily_vol * np.sqrt(horizon_days)
    expected_drift = (prob_up - 0.5) * 2 * sigma_horizon
    target_mid = latest_close * (1 + expected_drift)
    band_low = target_mid * (1 - sigma_horizon)
    band_high = target_mid * (1 + sigma_horizon)

    return {
        "horizon_days": horizon_days,
        "as_of": latest_date,
        "current_close": round(latest_close, 2),
        "direction": direction,
        "confidence_pct": round(confidence * 100, 1),
        "target_mid": round(target_mid, 2),
        "range_low": round(band_low, 2),
        "range_high": round(band_high, 2),
    }
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
    st.divider()
    st.subheader("🧠 Experiment Memory")
    st.caption(
        f"Logged runs in session: {len(st.session_state['experiment_history'])}"
    )
    if st.button("🗑️ Clear Experiment Memory"):
        st.session_state["experiment_history"] = []
        st.rerun()

# 2. Main UI Tabs
tab_run, tab_drive = st.tabs(["🧪 Experiment Loop", "☁️ Drive Artifacts"])

with tab_run:
    with st.expander(
        "📈 Configure Market Dataset & Forecast Horizon", expanded=True
    ):
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        ticker = d_col1.text_input(
            "Ticker Symbol (e.g., ^NSEI, RELIANCE.NS, SPY)",
            value="^NSEI",
        )
        period = d_col2.selectbox("History", ["5y", "10y", "max"], index=0)
        horizon_days = d_col3.selectbox(
            "Forecast Horizon (Trading Days)",
            options=[1, 5, 10, 20],
            index=1,
            format_func=lambda d: f"{d} Trading Day{'s' if d > 1 else ''}",
        )
        d_col4.write("")
        d_col4.write("")
        if d_col4.button("⬇️ Fetch Market Data"):
            with st.spinner(f"Downloading {ticker}..."):
                hist = yf.download(ticker, period=period, progress=False)
                if hist.empty:
                    st.error(f"No data returned for `{ticker}`.")
                else:
                    if isinstance(hist.columns, pd.MultiIndex):
                        hist.columns = hist.columns.get_level_values(0)
                    os.makedirs("data", exist_ok=True)
                    hist.reset_index().to_csv(
                        "data/market_data.csv", index=False
                    )
                    st.session_state["active_ticker"] = ticker
                    st.success(
                        f"Saved {len(hist)} trading days for **{ticker}** to `data/market_data.csv`!"
                    )

        if os.path.exists("data/market_data.csv"):
            cached_df = pd.read_csv("data/market_data.csv")
            active_sym = st.session_state.get("active_ticker", "^NSEI")
            st.caption(
                f"✅ Active local dataset: `{active_sym}` in `data/market_data.csv` ({len(cached_df)} rows) | Selected Forecast Window: **{horizon_days} Trading Day{'s' if horizon_days > 1 else ''}**"
            )
        else:
            st.caption(
                "ℹ️ No local CSV yet — click **Fetch Market Data** to download real market prices."
            )
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
        max_attempts = st.slider(
            "Max Autonomous Attempts per Run (stops early once baseline is beaten)",
            min_value=1,
            max_value=5,
            value=3,
        )

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

                approved_run = False
                for attempt in range(1, max_attempts + 1):
                    st.markdown(f"---\n### 🔄 Iteration {attempt}/{max_attempts}")
                    history_context = (
                        f"Baseline: {base_metrics}\nPast Attempts:\n"
                        + "\n".join(st.session_state["experiment_history"][-6:])
                    )

                    if groq_live:
                        st.write("2️⃣ Groq Researcher generating hypothesis from memory...")
                        hyp = groq_agent.generate_hypothesis(
                            baseline_code, history_context
                        )
                        st.info(f"**Hypothesis:** {hyp.title}\n\n{hyp.rationale}")

                        st.write("3️⃣ Groq Engineer writing candidate experiment...")
                        candidate_code = groq_agent.write_code(baseline_code, hyp)
                    else:
                        st.write("2️⃣ [Demo Mode] Generating sample hypothesis...")
                        hyp = HypothesisOutput(
                            title="Regularized Logistic Regression on Trend Ratios",
                            rationale="Replacing unregularized trees with L2-penalized LogisticRegression reduces overfitting on swing features.",
                            proposed_changes="Use StandardScaler + LogisticRegression(C=0.1, random_state=42).",
                        )
                        st.info(f"**Hypothesis:** {hyp.title}\n\n{hyp.rationale}")
                        candidate_code = baseline_code.replace(
                            "from sklearn.ensemble import RandomForestClassifier",
                            "from sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline",
                        ).replace(
                            "RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)",
                            "make_pipeline(StandardScaler(), LogisticRegression(C=0.1, random_state=42))",
                        )

                    os.makedirs("outputs", exist_ok=True)
                    cand_path = "outputs/candidate_experiment.py"
                    with open(cand_path, "w") as f:
                        f.write(candidate_code)

                    st.write("4️⃣ Running AST safety check & sandboxed execution...")
                    cand_res = execute_experiment(cand_path)

                    retry_count = 0
                    while not cand_res["success"] and groq_live and retry_count < 2:
                        retry_count += 1
                        st.warning(
                            f"⚠️ Runtime error detected. Auto-fixing (Debug {retry_count}/2)..."
                        )
                        candidate_code = groq_agent.fix_code(
                            candidate_code, cand_res["error"]
                        )
                        with open(cand_path, "w") as f:
                            f.write(candidate_code)
                        cand_res = execute_experiment(cand_path)

                    if not cand_res["success"]:
                        st.session_state["experiment_history"].append(
                            f"- [FAILED CODE] {hyp.title}: {cand_res['error'][:120]}"
                        )
                        st.error(f"Sandbox Execution Failed:\n{cand_res['error']}")
                        continue

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
                        ) > base_metrics.get("val_accuracy", 0)
                        critic = CriticOutput(
                            approved=improved,
                            feedback=(
                                "[Demo Mode] Out-of-sample swing accuracy improved."
                                if improved
                                else "[Demo Mode] Did not beat baseline out-of-sample."
                            ),
                        )

                    st.session_state["latest_diff"] = generate_unified_diff(
                        baseline_code, candidate_code
                    )
                    st.session_state["base_metrics"] = base_metrics
                    st.session_state["new_metrics"] = new_metrics
                    if critic.approved:
                        st.session_state["experiment_history"].append(
                            f"- [APPROVED] {hyp.title} -> {new_metrics}"
                        )
                        st.success(f"Approved by Critic: {critic.feedback}")
                        if groq_live:
                            paper_md = groq_agent.draft_paper(
                                hyp,
                                base_metrics,
                                new_metrics,
                                critic,
                                baseline_code=baseline_code,
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
                        st.session_state["latest_winner_code"] = candidate_code
                        approved_run = True
                        status.update(
                            label=f"Winning Hypothesis Found on Attempt {attempt}!",
                            state="complete",
                        )
                        break
                    else:
                        st.session_state["experiment_history"].append(
                            f"- [REJECTED] {hyp.title} -> {new_metrics} (Reason: {critic.feedback})"
                        )
                        st.warning(f"Rejected by Critic: {critic.feedback}")

                if not approved_run:
                    st.session_state.pop("latest_paper", None)
                    st.session_state.pop("latest_slug", None)
                    st.session_state.pop("latest_winner_code", None)
                    status.update(
                        label="Completed all attempts — baseline remains unbeaten.",
                        state="complete",
                    )

    # Dynamic Horizon Price Range & Signal Forecast Card
    if os.path.exists("data/market_data.csv"):
        st.divider()
        active_sym = st.session_state.get("active_ticker", "^NSEI")
        day_label = f"{horizon_days}-Trading-Day"
        st.subheader(f"🎯 Live {day_label} Market Forecast ({active_sym})")
        forecast = compute_live_forecast(
            "data/market_data.csv", horizon_days=horizon_days
        )
        if forecast:
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            f_col1.metric(
                label=f"Latest Close ({forecast['as_of'][:10]})",
                value=f"{forecast['current_close']:,.2f}",
            )
            f_col2.metric(
                label=f"{horizon_days}D AI Signal",
                value=forecast["direction"],
                delta=f"{forecast['confidence_pct']}% Confidence",
            )
            f_col3.metric(
                label=f"{horizon_days}D Expected Target",
                value=f"{forecast['target_mid']:,.2f}",
                delta=f"{forecast['target_mid'] - forecast['current_close']:+,.2f}",
            )
            f_col4.metric(
                label=f"{horizon_days}D Projected Range",
                value=f"{forecast['range_low']:,.0f} – {forecast['range_high']:,.0f}",
            )
            st.caption(
                f"Note: Range reflects a ±1σ {horizon_days}-trading-day volatility envelope derived from historical returns. For research and educational purposes only."
            )

    if "latest_diff" in st.session_state:
        st.divider()
        vis_col1, vis_col2 = st.columns(2)
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

        dl_col1, dl_col2, dl_col3 = st.columns(3)
        slug = st.session_state.get("latest_slug", "experiment")

        with dl_col1:
            st.download_button(
                label="⬇️ Download Research Paper (.md)",
                data=st.session_state["latest_paper"],
                file_name=f"{slug}.md",
                mime="text/markdown",
            )
        with dl_col2:
            if "latest_winner_code" in st.session_state:
                st.download_button(
                    label="⬇️ Download Winning Model (.py)",
                    data=st.session_state["latest_winner_code"],
                    file_name=f"{slug}.py",
                    mime="text/x-python",
                )
        with dl_col3:
            if st.button("🌐 Publish Report to srinivasta.github.io"):
                if gh_live:
                    live_url = gh_mgr.publish_to_portfolio(
                        slug,
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
