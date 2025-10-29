"""
GitHub API integration for triggering workflows.
"""

import logging
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


class GitHubWorkflowTrigger:
    """
    Trigger GitHub Actions workflows via API.
    """

    def __init__(self, token: Optional[str], repo: str):
        """
        Initialize GitHub workflow trigger.

        Args:
            token: GitHub Personal Access Token
            repo: Repository in format 'owner/repo'
        """
        self.token = token
        self.repo = repo
        self.base_url = "https://api.github.com"

        if not self.token:
            logger.warning("GitHub token not configured - workflow triggering disabled")

    def trigger_workflow(
        self, workflow_id: str, ref: str = "main", inputs: Optional[Dict] = None
    ) -> Dict:
        """
        Trigger a GitHub Actions workflow.

        Args:
            workflow_id: Workflow filename (e.g., 'automated-retraining.yml')
            ref: Git reference (branch/tag) to run workflow on
            inputs: Optional workflow inputs

        Returns:
            Dictionary with trigger result

        Raises:
            Exception: If trigger fails
        """
        if not self.token:
            return {
                "success": False,
                "error": "GitHub token not configured",
                "message": "Set GITHUB_TOKEN environment variable to enable automatic triggering",
            }

        url = f"{self.base_url}/repos/{self.repo}/actions/workflows/{workflow_id}/dispatches"

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        payload = {"ref": ref}

        if inputs:
            payload["inputs"] = inputs

        logger.info(f"Triggering workflow: {workflow_id} on {ref}")

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)

            if response.status_code == 204:
                logger.info(f"✅ Workflow triggered successfully: {workflow_id}")
                return {
                    "success": True,
                    "message": "Workflow triggered successfully",
                    "workflow": workflow_id,
                    "ref": ref,
                    "url": f"https://github.com/{self.repo}/actions",
                }
            else:
                error_msg = response.text
                logger.error(
                    f"Failed to trigger workflow: {response.status_code} - {error_msg}"
                )
                return {
                    "success": False,
                    "error": f"GitHub API error: {response.status_code}",
                    "message": error_msg,
                }

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": "Request failed",
                "message": str(e),
            }

    def get_workflow_status(self, workflow_id: str, limit: int = 5) -> Dict:
        """
        Get recent runs of a workflow.

        Args:
            workflow_id: Workflow filename
            limit: Number of recent runs to fetch

        Returns:
            Dictionary with workflow run information
        """
        if not self.token:
            return {
                "success": False,
                "error": "GitHub token not configured",
            }

        url = f"{self.base_url}/repos/{self.repo}/actions/workflows/{workflow_id}/runs"

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        params = {"per_page": limit}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                runs = []

                for run in data.get("workflow_runs", [])[:limit]:
                    runs.append(
                        {
                            "id": run["id"],
                            "status": run["status"],
                            "conclusion": run.get("conclusion"),
                            "created_at": run["created_at"],
                            "updated_at": run["updated_at"],
                            "html_url": run["html_url"],
                            "run_number": run["run_number"],
                        }
                    )

                return {
                    "success": True,
                    "total_count": data.get("total_count", 0),
                    "runs": runs,
                }
            else:
                logger.error(f"Failed to get workflow status: {response.status_code}")
                return {
                    "success": False,
                    "error": f"GitHub API error: {response.status_code}",
                }

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
