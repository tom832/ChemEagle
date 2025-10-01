def plan_observer_agent(extraction_plan: dict) -> dict:

    steps = extraction_plan.get("steps", [])
    errors = []
    for idx, step in enumerate(steps):
        if not step.get("agent") or not step.get("task"):
            errors.append(f"Step {idx} missing agent or task.")

    result = {
        "plan_reviewed": True,
        "errors": errors,
        "plan_summary": f"Plan contains {len(steps)} steps; {len(errors)} errors found."
    }
    return result


def action_observer_agent(agent_actions: list) -> dict:

    issues = []
    for idx, action in enumerate(agent_actions):
        if action.get("status") != "success":
            issues.append(f"Action {idx} failed: {action.get('error', 'Unknown error')}")

    review = {
        "actions_reviewed": True,
        "issues": issues,
        "actions_summary": f"{len(agent_actions)} actions checked; {len(issues)} issues found."
    }
    return review
