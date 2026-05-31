"""
Multi-Agent Underwriting System
Specialized agents orchestrated by LangGraph for insurance risk assessment.
"""

from typing import Annotated, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import json


# ── State ──────────────────────────────────────────────────────────────────────

class UnderwritingState(TypedDict):
    applicant_data: dict
    messages: Annotated[list, add_messages]
    research_findings: str
    risk_score: float
    risk_factors: list[str]
    compliance_flags: list[str]
    compliance_pass: bool
    final_report: dict
    next_agent: str


# ── Output schemas ─────────────────────────────────────────────────────────────

class RiskAssessment(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Risk score from 0 (low) to 1 (high)")
    factors: list[str] = Field(description="Key risk factors identified")
    recommendation: Literal["approve", "refer", "decline"]
    reasoning: str


class ComplianceCheck(BaseModel):
    passed: bool
    flags: list[str] = Field(description="Any compliance issues found")
    notes: str


# ── Agents ─────────────────────────────────────────────────────────────────────

def make_llm(model: str = "gpt-4o") -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=0)


def research_agent(state: UnderwritingState) -> UnderwritingState:
    """Gathers and summarizes relevant background on the applicant."""
    llm = make_llm()
    applicant = state["applicant_data"]

    response = llm.invoke([
        SystemMessage(content=(
            "You are an insurance research agent. Summarize the applicant's profile "
            "and identify any background factors relevant to underwriting. Be factual and concise."
        )),
        HumanMessage(content=f"Applicant data:\n{json.dumps(applicant, indent=2)}")
    ])

    return {**state, "research_findings": response.content, "next_agent": "risk_assessment"}


def risk_assessment_agent(state: UnderwritingState) -> UnderwritingState:
    """Scores risk and identifies key risk factors."""
    llm = make_llm().with_structured_output(RiskAssessment)
    applicant = state["applicant_data"]
    findings = state["research_findings"]

    assessment: RiskAssessment = llm.invoke([
        SystemMessage(content=(
            "You are an insurance risk assessment specialist. "
            "Evaluate the applicant and return a structured risk assessment."
        )),
        HumanMessage(content=(
            f"Research findings:\n{findings}\n\n"
            f"Raw applicant data:\n{json.dumps(applicant, indent=2)}"
        ))
    ])

    return {
        **state,
        "risk_score": assessment.score,
        "risk_factors": assessment.factors,
        "next_agent": "compliance"
    }


def compliance_agent(state: UnderwritingState) -> UnderwritingState:
    """Checks for regulatory and policy compliance issues."""
    llm = make_llm().with_structured_output(ComplianceCheck)
    applicant = state["applicant_data"]

    check: ComplianceCheck = llm.invoke([
        SystemMessage(content=(
            "You are an insurance compliance officer. Check the applicant data "
            "for any regulatory, sanctions, or policy compliance issues."
        )),
        HumanMessage(content=f"Applicant data:\n{json.dumps(applicant, indent=2)}")
    ])

    return {
        **state,
        "compliance_flags": check.flags,
        "compliance_pass": check.passed,
        "next_agent": "report_generation"
    }


def report_generation_agent(state: UnderwritingState) -> UnderwritingState:
    """Synthesizes all agent findings into a structured underwriting report."""
    llm = make_llm()

    summary_prompt = (
        f"Generate a structured underwriting report based on the following:\n\n"
        f"Applicant: {json.dumps(state['applicant_data'], indent=2)}\n\n"
        f"Research findings: {state['research_findings']}\n\n"
        f"Risk score: {state['risk_score']:.2f} | Risk factors: {state['risk_factors']}\n\n"
        f"Compliance passed: {state['compliance_pass']} | Flags: {state['compliance_flags']}\n\n"
        f"Provide: executive summary, risk assessment, compliance status, and final recommendation."
    )

    response = llm.invoke([
        SystemMessage(content="You are a senior underwriter. Write a professional underwriting report."),
        HumanMessage(content=summary_prompt)
    ])

    report = {
        "risk_score": state["risk_score"],
        "risk_factors": state["risk_factors"],
        "compliance_passed": state["compliance_pass"],
        "compliance_flags": state["compliance_flags"],
        "recommendation": "approve" if state["risk_score"] < 0.5 and state["compliance_pass"] else "refer",
        "narrative": response.content,
    }

    return {**state, "final_report": report, "next_agent": END}


def route_agent(state: UnderwritingState) -> str:
    return state.get("next_agent", END)


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_underwriting_graph() -> StateGraph:
    graph = StateGraph(UnderwritingState)

    graph.add_node("research", research_agent)
    graph.add_node("risk_assessment", risk_assessment_agent)
    graph.add_node("compliance", compliance_agent)
    graph.add_node("report_generation", report_generation_agent)

    graph.set_entry_point("research")
    graph.add_conditional_edges("research", route_agent)
    graph.add_conditional_edges("risk_assessment", route_agent)
    graph.add_conditional_edges("compliance", route_agent)
    graph.add_edge("report_generation", END)

    return graph.compile()


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_underwriting_graph()

    sample_applicant = {
        "name": "Acme Logistics LLC",
        "industry": "Commercial Transportation",
        "years_in_business": 8,
        "annual_revenue": 4200000,
        "num_vehicles": 12,
        "prior_claims": [
            {"year": 2022, "type": "collision", "amount": 15000},
            {"year": 2023, "type": "cargo_damage", "amount": 8500},
        ],
        "coverage_requested": "Commercial Auto + General Liability",
        "coverage_limit_requested": 2000000,
    }

    initial_state = UnderwritingState(
        applicant_data=sample_applicant,
        messages=[],
        research_findings="",
        risk_score=0.0,
        risk_factors=[],
        compliance_flags=[],
        compliance_pass=True,
        final_report={},
        next_agent="research",
    )

    print("Running multi-agent underwriting...\n")
    result = app.invoke(initial_state)

    print("=" * 60)
    print("UNDERWRITING REPORT")
    print("=" * 60)
    report = result["final_report"]
    print(f"Risk Score:       {report['risk_score']:.2f}")
    print(f"Recommendation:   {report['recommendation'].upper()}")
    print(f"Compliance:       {'PASS' if report['compliance_passed'] else 'FAIL'}")
    if report["compliance_flags"]:
        print(f"Flags:            {', '.join(report['compliance_flags'])}")
    print(f"\nNarrative:\n{report['narrative']}")
