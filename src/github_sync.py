from datetime import datetime
from github import Github


class GitHubManager:

    def __init__(self, token: str, repo_name: str, pages_repo: str):
        self.gh = Github(token)
        self.repo = self.gh.get_repo(repo_name)
        self.pages_repo = self.gh.get_repo(pages_repo)

    def create_experiment_pr(
        self,
        branch_name: str,
        new_code: str,
        report_md: str,
        title: str,
        body: str,
    ) -> str:
        main_branch = self.repo.get_branch("main")
        self.repo.create_git_ref(
            ref=f"refs/heads/{branch_name}", sha=main_branch.commit.sha
        )

        # Update baseline script in the new branch
        contents = self.repo.get_contents(
            "baselines/baseline_experiment.py", ref=branch_name
        )
        self.repo.update_file(
            path=contents.path,
            message=f"feat(exp): {title}",
            content=new_code,
            sha=contents.sha,
            branch=branch_name,
        )

        # Commit research paper draft
        report_path = f"reports/{branch_name}.md"
        self.repo.create_file(
            path=report_path,
            message=f"docs(paper): add report for {title}",
            content=report_md,
            branch=branch_name,
        )

        pr = self.repo.create_pull(
            title=f"[AI Scientist] {title}",
            body=body,
            head=branch_name,
            base="main",
        )
        return pr.html_url

    def publish_to_portfolio(self, slug: str, markdown_content: str) -> str:
        """Publishes approved research markdown directly to srinivasta.github.io."""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        file_path = f"_posts/{date_str}-{slug}.md"
        try:
            existing = self.pages_repo.get_contents(file_path, ref="main")
            self.pages_repo.update_file(
                file_path,
                f"Update AI research post: {slug}",
                markdown_content,
                existing.sha,
                branch="main",
            )
        except Exception:
            self.pages_repo.create_file(
                file_path,
                f"Publish AI research post: {slug}",
                markdown_content,
                branch="main",
            )
        return f"https://github.io{date_str}/{slug}/"
