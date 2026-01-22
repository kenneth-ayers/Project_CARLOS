# Project_CARLOS
Project CARLOS is a human-scale companion robotics research platform focused on safe behavior design, failure containment, and reliable operation in close proximity to humans. The project explores a layered robotic control architecture emphasizing predictable state transitions, degraded-mode operation, and conservative autonomy boundaries rather than performance-driven optimization.

The system is designed under a “when, not if” failure philosophy, assuming sensor uncertainty, communication loss, and human misuse as normal operating conditions. Core development focuses on behavior-level autonomy implemented through explicit state machines, supervised control layers, and safe default states that prioritize human safety and system stability over task completion.

Mechanical design emphasizes low-energy actuation, compliant interaction zones, and intentional limitations on speed and force. Electrical and control subsystems prioritize observability, fault detection, and controlled shutdown behavior. A formal Failure Modes and Effects Analysis (FMEA) is used throughout development to identify high-risk interactions and guide mitigation strategies at both the hardware and software levels.

Project CARLOS is not intended as a demonstration of general intelligence, but as an investigation into responsible robot behavior under uncertainty. The platform serves as a testbed for studying how companion robots can behave transparently, predictably, and safely in human environments when autonomy is partial, assumptions fail, or operator intent is ambiguous.
Prior Art and Defensive Disclosure

This project is accompanied by a formal defensive publication establishing prior art for the core concepts and architectural principles described herein.

Defensive Disclosure:
CARLOS – Defensive Technical Disclosure
Published via Zenodo (CERN)
DOI: [insert Zenodo DOI here once published]

The disclosure documents the system architecture, behavioral design philosophy, safety constraints, and obvious technical variations of the CARLOS companion robotics platform. It is intended to ensure that these foundational concepts remain part of the public commons and are not subject to future patent claims.

This repository and all associated materials are released under the GNU Affero General Public License v3.0 (AGPL-3.0).
