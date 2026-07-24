import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


class GithubIssueCreator:

    def __init__(
            self,
            notification_threshold_minutes: int = 30,
            do_github_testing: bool = False,
            ):

        pw_file = Path(__file__).parent.parent.parent.parent / ".json"
        with open(pw_file, encoding="utf-8") as f:
            self._config = json.load(f)

        self._notification_threshold = pd.Timedelta(minutes=notification_threshold_minutes)
        self._exception_since = None
        self._netload_mismatch_since = None

        if do_github_testing:
            self._create_issue(
                title="[Home BESS] GitHub issue creator test",
                body="This is a test issue created to verify GitHub integration."
            )

    def on_loop_success(self) -> None:
        """Reset exception tracking after a successful loop iteration."""
        self._exception_since = None

    def handle_loop_exception(self, error: Exception) -> None:
        """Track loop exceptions and create a GitHub issue after the threshold."""
        now = pd.Timestamp.now(tz="Europe/Berlin")
        if self._exception_since is None:
            self._exception_since = now
        elif now - self._exception_since >= self._notification_threshold:
            try:
                self._create_issue(
                    title="[Home BESS] MPC Controller errors for >30 min",
                    body=(
                        f"An error occurred in the MPC main loop at {now}:\n\n"
                        f"{type(error).__name__}:\n\n"
                        f"Errors ongoing since {self._exception_since}."
                    ),
                )
                # Set to high value to avoid repeated issues
                self._exception_since = now + pd.Timedelta(days=365)
            except Exception as gh_err:
                print(f"Failed to create GitHub issue: {gh_err}")

    def handle_netload_check(
        self, set_netload_kw: float, act_netload_kw: float, current_time: pd.Timestamp
    ) -> None:
        """Check net load mismatch and create a GitHub issue after the threshold."""
        if not np.isclose(act_netload_kw, set_netload_kw, rtol=1e-2):
            print(
                f"Warning: Set net load {set_netload_kw:.2f} kW does not match "
                f"actual net load {act_netload_kw:.2f} kW."
            )
            if self._netload_mismatch_since is None:
                self._netload_mismatch_since = current_time
            elif current_time - self._netload_mismatch_since >= self._notification_threshold:
                self._create_issue(
                    title="[Home BESS] Net load mismatch for >30 min",
                    body=(
                        f"Set net load {set_netload_kw:.2f} kW does not match "
                        f"actual net load {act_netload_kw:.2f} kW. "
                        f"Mismatch ongoing since {self._netload_mismatch_since}."
                    ),
                )
                # Set to high value to avoid repeated issues
                self._netload_mismatch_since = current_time + pd.Timedelta(days=365)
        else:
            self._netload_mismatch_since = None

    def _create_issue(self, title: str, body: str) -> None:
        """Create a GitHub issue with the given title and body."""
        owner = self._config["github_owner"]
        repo = self._config["github_repo"]
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        headers = {
            "Authorization": f"Bearer {self._config['github_token']}",
            "Accept": "application/vnd.github+json",
        }

        full_body = f"@molu-fhv\n\n{body}".strip()
        assignees = self._config.get("github_assignees", [])

        r = requests.post(url, headers=headers, json={"title": title, "body": full_body, "assignees": assignees}, timeout=10)

        if r.status_code == 201:
            print(f"GitHub issue created: {title}")
        else:
            print(f"Creating GitHub issue failed ({r.status_code}): {r.text}")
