from quant_platform.orchestration.corrective_wizard_browser_auth import (
    build_wizard_browser_auth_readiness,
)


if __name__ == "__main__":
    result = build_wizard_browser_auth_readiness()
    print(result.summary["status"])
    print(result.paths["status"])
