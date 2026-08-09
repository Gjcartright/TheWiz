from quant_platform.orchestration.state import OrchestratorState, StageResult, StageStatus
from quant_platform.orchestration.contracts import (
    AgentCapability,
    CandidateIdentity,
    ComparisonEvent,
    DecisionBucket,
    DecisionRecord,
    EvidencePacket,
    OutcomeRecord,
    TaskCard,
    TaskStatus,
    VetoRecord,
)
from quant_platform.orchestration.teacher_contracts import (
    CouncilContext,
    CouncilDecision,
    CouncilStatus,
    CriticAssessment,
    CriticType,
    CriticVerdict,
    EvidenceAuthority,
    ExactMode,
    StudentOutcomeForecast,
    StudentRouterPrediction,
    TeacherAction,
    TeacherProposal,
)

__all__ = [
    "OrchestratorState",
    "StageResult",
    "StageStatus",
    "AgentCapability",
    "CandidateIdentity",
    "ComparisonEvent",
    "DecisionBucket",
    "DecisionRecord",
    "EvidencePacket",
    "OutcomeRecord",
    "TaskCard",
    "TaskStatus",
    "VetoRecord",
    "CouncilContext",
    "CouncilDecision",
    "CouncilStatus",
    "CriticAssessment",
    "CriticType",
    "CriticVerdict",
    "EvidenceAuthority",
    "ExactMode",
    "StudentOutcomeForecast",
    "StudentRouterPrediction",
    "TeacherAction",
    "TeacherProposal",
    "run_langgraph_agent_workflow",
    "run_orchestrator",
]


def __getattr__(name: str):
    if name == "run_orchestrator":
        from quant_platform.orchestration.orchestrator import run_orchestrator

        return run_orchestrator
    if name == "run_langgraph_agent_workflow":
        from quant_platform.orchestration.langgraph_workflow import run_langgraph_agent_workflow

        return run_langgraph_agent_workflow
    raise AttributeError(name)
