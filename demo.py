"""
Jev System 1 Demonstration
==========================

Shows how programmatic code integrates with TypeSafe AI's Jev model
to make instant, type-safe, calibrated semantic judgments.
"""

from __future__ import annotations

import json
from jev_client import DecisionEngine, Choice, Score, Noul


def main() -> None:
    print("=" * 70)
    print(" Jev (TypeSafe AI) - System 1 Semantic Decision Demonstration")
    print("=" * 70)

    engine = DecisionEngine()
    if engine.has_active_credentials:
        print("[Status] Connected to live TypeSafe AI API using TYPESAFE_API_KEY.")
    else:
        print("[Status] Running in local simulation mode.")
        print("         To connect to live Jev inference, set TYPESAFE_API_KEY in .env")
        print("         Obtain your key from: https://console.typesafe.ai")
    print("-" * 70)

    # 1. Define the input state (unstructured telemetry / ticket)
    incident_state = {
        "event_id": "INC-89211",
        "timestamp": "2026-09-18T04:15:00Z",
        "reporter": "alex.dev@corp.internal",
        "raw_text": (
            "CRITICAL: Primary PostgreSQL replica is desynced by >400GB. "
            "Write latencies on auth services spiking past 12,000ms. "
            "Users in EU-West reporting complete authentication timeouts. "
            "Need immediate DBA on-call intervention."
        ),
    }

    print("\n[Input State]:")
    print(json.dumps(incident_state, indent=2))

    # 2. Define the schema of typed questions
    questions = {
        "is_critical_outage": Noul(
            instructions="Does this describe a critical service outage affecting production users?",
        ),
        "incident_domain": Choice(
            instructions="Which engineering domain is primarily responsible for resolving this issue?",
            criteria={
                "database_infra": "PostgreSQL, replication, storage, or I/O bottlenecks",
                "application_logic": "Software bugs, application exceptions, or bad deploy",
                "security_incident": "Intrusion, unauthorized access, or credential leak",
                "billing_service": "Payment gateway or billing transaction failure",
            },
        ),
        "severity_tier": Score(
            instructions="Assess the operational severity tier.",
            criteria=[
                "SEV-3: Low impact / non-blocking",
                "SEV-2: Degraded performance / partial impact",
                "SEV-1: Critical outage / major revenue or user impact",
            ],
        ),
    }

    print("\n[Evaluating State with Jev in a single parallel forward pass...]")
    result = engine.evaluate(incident_state, questions)

    print("\n" + result.summary())

    # 3. Demonstrate programmatic decision gating (The "Smart If Statement")
    print("\n" + "=" * 70)
    print(" Programmatic Decision Logic (Calibrated Risk Gating)")
    print("=" * 70)

    outage_prob = result.nouls["is_critical_outage"].noul
    primary_domain = result.choices["incident_domain"].choice
    domain_confidence = result.choices["incident_domain"].confidence
    severity = result.scores["severity_tier"].score

    print(f"-> P(Critical Outage)     : {outage_prob * 100:.1f}%")
    print(f"-> Domain Classification  : '{primary_domain}' (Confidence: {domain_confidence * 100:.1f}%)")
    print(f"-> Severity Score         : {severity:.2f} / 3.0")

    print("\n[Execution Action]:")
    if outage_prob >= 0.80 and severity >= 2.0:
        print(f"  [TRIGGER] Auto-escalating to #{primary_domain}-oncall via PagerDuty.")
        print(f"  [REASON] High calibrated confidence ({outage_prob:.2f}) and SEV-{round(severity)} impact.")
    elif outage_prob >= 0.40:
        print("  [ROUTER] Ambiguity detected in System 1 pass. Escalating to human triage desk.")
    else:
        print("  [RESOLVE] Normal priority. Routed to standard asynchronous ticket queue.")

    print("\n" + "=" * 70)
    print(" Demonstration complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
