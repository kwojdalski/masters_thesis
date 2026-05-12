"""Auto Issue Triage skill - Automatically pick up and work on most severe GitHub issues."""

import subprocess
import sys
from pathlib import Path


def get_highest_priority_issues(limit: int = 5) -> list[dict]:
    """Get highest priority open issues sorted by severity and age.

    Args:
        limit: Maximum number of issues to return

    Returns:
        List of issues sorted by priority (critical → high → medium → low)
    """
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--state", "open", "--limit", str(limit), "--json"],
            capture_output=True,
            text=True,
            check=True
        )

        if result.returncode != 0:
            print(f"Error: Failed to fetch issues: {result.stderr}")
            return []

        import json
        issues = json.loads(result.stdout)

        # Filter only open issues
        open_issues = [issue for issue in issues if issue.get("state") == "open"]

        if not open_issues:
            print("No open issues found to triage.")
            return []

        # Sort by severity priority and then by age (newest first)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, None: 4}

        sorted_issues = sorted(
            open_issues,
            key=lambda x: (
                severity_order.get(x.get("labels", {}).get(0, "none"), 4)  # Priority by severity
                x.get("created_at", "")
            )
        )

        # Take top N issues
        return sorted_issues[:limit]


def analyze_issue(issue: dict) -> dict:
    """Analyze whether an issue is a real bug vs documentation/feature request.

    Args:
        issue: GitHub issue object

    Returns:
        Analysis dict with verdict and evidence
    """
    issue_number = issue.get("number", "")
    title = issue.get("title", "")
    body = issue.get("body", "")
    labels = issue.get("labels", [])

    analysis = {
        "issue_number": issue_number,
        "title": title,
        "is_real_issue": False,
        "evidence": [],
        "verdict": "NEEDS_INVESTIGATION",
        "next_actions": []
    }

    # Heuristic analysis for real issues
    real_issue_indicators = [
        "bug", "error", "exception", "crash", "broken", "fails",
        "fix", "correct", "resolve", "patch", "improve",
        "optimization", "refactor", "update", "upgrade"
    ]

    # Check for issue description patterns
    title_lower = title.lower()
    body_lower = body.lower()

    # Indicators this might be a real issue
    has_action_verbs = any(
        word in title_lower for word in real_issue_indicators
    )
    has_error_terms = any(
        word in title_lower + " " + body_lower
        for word in ["error", "exception", "crash", "fails", "bug", "fix"]
    )

    # Documentation vs real issue detection
    doc_indicators = [
        "documentation", "doc", "readme", "changelog", "guide", "tutorial",
        "example", "clarification", "update", "typo", "format", "style"
    ]

    has_doc_requests = any(
        word in title_lower + " " + body_lower
        for word in doc_indicators
    )

    # Make初步 verdict
    if has_action_verbs and has_error_terms:
        # Looks like a real bug or fix request
        analysis["is_real_issue"] = True
        analysis["verdict"] = "CONFIRMED"
        analysis["next_actions"] = [
            "Investigate code location mentioned in issue",
            "Verify bug with existing tests",
            "Create reproduction case if possible",
            "Check git log for recent related changes"
        ]
    elif has_doc_requests:
        # Looks like documentation request
        analysis["verdict"] = "ALREADY_DOCUMENTED"
        analysis["is_real_issue"] = False
        analysis["next_actions"] = [
            "Check if documentation exists in codebase",
            "Consider adding to existing docs",
            "Update docstrings if needed"
        ]
    else:
        # Need more investigation
        analysis["verdict"] = "NEEDS_INVESTIGATION"
        analysis["next_actions"] = [
            "Read issue body and comments thoroughly",
            "Search codebase for relevant code",
            "Ask user for clarification if needed"
        ]

    analysis["evidence"] = {
        "title": title,
        "has_action_verbs": has_action_verbs,
        "has_error_terms": has_error_terms,
        "has_doc_requests": has_doc_requests,
        "labels": labels
    }

    return analysis


def verify_codebase_claim(issue: dict, issue_analysis: dict) -> dict:
    """Verify if issue claims match actual codebase behavior.

    Args:
        issue: GitHub issue object
        issue_analysis: Analysis from analyze_issue()

    Returns:
        Verification results with evidence
    """
    title = issue.get("title", "")
    body = issue.get("body", "")
    issue_number = issue.get("number", "")

    verification = {
        "issue_number": issue_number,
        "claim_matches_code": False,
        "evidence": [],
        "verdict": "",
        "additional_findings": []
    }

    # Look for file paths in issue
    import re
    file_pattern = r'\`[^`]+\.[a-z]{2,4}`:[0-9]+'

    def find_all_occurrences(text: str, pattern: str) -> list[str]:
        """Find all occurrences of a pattern in text."""
        return [m.start() for m in re.finditer(pattern, text)]

    title_text = title + " " + body
    file_matches = find_all_occurrences(title_text, file_pattern)

    if file_matches:
        verification["claim_matches_code"] = True
        verification["evidence"].append(
            f"Found file reference: {len(file_matches)} occurrence(s)"
        )

        # Try to locate the files
        for match in file_matches[:3]:  # Check up to 3 most relevant mentions
            file_path = match.group(1)  # Extract file path from backticks
            file_path_obj = Path(file_path)

            if file_path_obj.exists():
                verification["evidence"].append(
                    f"File exists: {file_path_obj.absolute()}"
                )
            else:
                verification["evidence"].append(
                    f"File not found: {file_path_obj.absolute()}"
                )
                break

    # Look for function mentions
    func_pattern = r'\b[a-z_]+\(|\s+[a-z_]+\)'

    def find_functions(text: str) -> list[str]:
        """Find function names in text."""
        matches = re.findall(r'\b[a-z_]+\(|\s+[a-z_]+\)', text)
        return [m for m in matches if len(m.group(1)) > 2]  # Filter out short ones

    title_funcs = find_functions(title)
    body_funcs = find_functions(body)

    if title_funcs:
        verification["evidence"].append(
            f"Functions found in title: {', '.join(title_funcs[:3])}"
        )

    if body_funcs:
        verification["evidence"].append(
            f"Functions found in body: {', '.join(body_funcs[:3])}"
        )

    # Determine verdict based on verification
    if verification["claim_matches_code"]:
        if file_matches and len([m for m in file_matches if Path(m.group(1)).exists()]) > 0:
            verification["verdict"] = "CONFIRMED"
            verification["additional_findings"].append(
                "At least one referenced file exists in codebase"
            )
        else:
            verification["verdict"] = "NOT_CONFIRMED"
            verification["additional_findings"].append(
                "Referenced files not found or codebase structure differs"
            )

    verification["evidence"] = verification["evidence"]
    return verification


def generate_triage_report(issues: list[dict]) -> str:
    """Generate detailed triage report."""

    if not issues:
        return "No open issues found requiring triage."

    report = []
    report.append("# Auto Issue Triage Report")
    report.append(f"Generated: 2026-05-12")
    report.append("")
    for i, issue in enumerate(issues, 1):
        issue_number = issue.get("number", i)
        report.append(f"## Issue #{issue_number}: {issue.get('title', 'Unknown Issue')}")
        report.append("")

        # Get issue details
        analysis = analyze_issue(issue)
        verdict = analysis["verdict"]
        labels = issue.get("labels", [])
        created_at = issue.get("created_at", "")

        report.append(f"**Status:** {verdict}")
        report.append(f"**Labels:** {', '.join(labels) if labels else 'none'}")
        report.append(f"**Severity:** {get_severity_display(labels)}")
        report.append(f"**Created:** {created_at}")
        report.append(f"**URL:** {issue.get('html_url', 'N/A')}")
        report.append("")

        # Evidence section
        if analysis["evidence"]:
            report.append("**Evidence Found:**")
            for evidence in analysis["evidence"][:5]:  # Show top 5 items
                report.append(f"  • {evidence}")
        else:
            report.append("No code evidence found in issue description")

        report.append("")
        # Analysis section
        report.append("**Analysis:**")
        if analysis["is_real_issue"]:
            report.append("This appears to be a real bug or fix request.")
            if analysis["next_actions"]:
                report.append("**Recommended Actions:**")
                for action in analysis["next_actions"]:
                    report.append(f"  • {action}")
        elif analysis["verdict"] == "ALREADY_DOCUMENTED":
            report.append("This appears to be a documentation or clarification request.")
            if analysis["next_actions"]:
                report.append("**Recommended Actions:**")
                for action in analysis["next_actions"]:
                    report.append(f"  • {action}")
        else:
            report.append("This issue needs more investigation before taking action.")
            if analysis["next_actions"]:
                report.append("**Recommended Actions:**")
                for action in analysis["next_actions"]:
                    report.append(f"  • {action}")

        report.append("")
        # Next steps section
        report.append("**Next Steps:**")
        if verdict == "CONFIRMED":
            report.append("1. Add confirmation label: `gh issue edit {issue_number} --add-label \"triaged-confirmed\"")
            report.append("2. Start working on fix")
            report.append("3. Create feature branch: `git checkout -b fix/issue-{issue_number}\"")
        elif verdict == "ALREADY_DOCUMENTED":
            report.append("1. Add clarification comment if needed")
            report.append("2. Consider updating documentation if applicable")
        else:
            report.append("1. Request more information from issue author")
            report.append("2. Add labels for better categorization")
            report.append("3. Wait for user confirmation before taking action")

        report.append("")
        report.append("---")

    return "\n".join(report)


def get_severity_display(labels: list) -> str:
    """Convert labels to severity display."""
    severity_map = {
        "critical": "Critical",
        "high": "High",
        "medium": "Medium",
        "low": "Low"
    }

    for label in labels:
        if label.lower() in severity_map:
            return severity_map[label.lower()]
    return "Unspecified"


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        try:
            max_issues = int(sys.argv[1])
        except ValueError:
            print("Error: Max issues must be a positive integer.")
            print("Usage: python auto_issue_triage.py [max_issues]")
            return
    else:
        max_issues = 5

    print(f"Auto Issue Triage - Checking up to {max_issues} highest priority open issues...")

    issues = get_highest_priority_issues(max_issues)

    if not issues:
        print("No open issues found requiring triage.")
        print("All issues resolved! Skill is no longer needed.")
        return

    print(f"Found {len(issues)} issue(s) to triage")
    print("")

    # Generate analysis for each issue
    all_analyses = []
    for issue in issues:
        analysis = analyze_issue(issue)
        verification = verify_codebase_claim(issue, analysis)

        all_analyses.append({
            "issue": issue,
            "analysis": analysis,
            "verification": verification
        })

    # Sort by priority for display
    sorted_issues = sorted(all_analyses, key=lambda x: (
        {"CONFIRMED": 0, "ALREADY_DOCUMENTED": 1, "NEEDS_INVESTIGATION": 2, "NOT_CONFIRMED": 3}
    )

    # Generate report
    report = generate_triage_report([item["issue"] for item in sorted_issues])

    print(report)

    # Optional: Save report to file
    report_file = Path("auto_triage_report.md")
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_file}")
    print("")
    print("## Actions Available")
    print("To work on the top issues interactively:")
    for i, item in enumerate(sorted_issues[:3], 1):
        issue_number = item["issue"].get("number", "")
        verdict = item["analysis"]["verdict"]
        print(f"  {i+1}. gh issue view {issue_number} -- View full details")
        print(f"  {i+1}. gh issue edit {issue_number} --add-label \"in-progress\"")
        print(f"  {i+1}. git checkout -b fix/issue-{issue_number} -- Create branch for this issue")

    print("\nTo use automatic mode:")
    print("  ds-triage --max-issues <n>     — Auto-triage N issues")
    print("  ds-triage --dry-run                    — Preview actions without making changes")
    print("  ds-triage --severity <level>         — Only process issues at or above severity")


if __name__ == "__main__":
    main()
