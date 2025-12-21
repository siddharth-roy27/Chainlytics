# CHAINLYTICS — BACKEND ML SYSTEM

## Overview

Chainlytics is a comprehensive backend ML system for logistics decision making. It provides a complete suite of machine learning services for demand forecasting, scenario generation, cost modeling, state encoding, policy optimization, routing, safety enforcement, anomaly detection, and explainability.

## System Architecture

```
chainlytics-backend/
│
├── services/
│   ├── forecast_service/         # Demand forecasting (P50/P90/P99)
│   ├── scenario_service/         # Scenario & risk generation
│   ├── cost_service/             # Price & cost modeling
│   ├── state_service/            # Graph state encoding
│   ├── policy_service/           # Multi-agent RL core
│   ├── routing_service/          # Route optimization
│   ├── safety_service/           # Constraint engine
│   ├── anomaly_service/          # Monitoring & alerts
│   └── explain_service/          # Explainability
│
├── simulation/
│   ├── env/                      # Base simulation environment
│   ├── dynamics/                 # Order & vehicle dynamics
│   ├── reward/                   # Reward functions (v1/v2)
│   ├── constraints/              # Constraint checking
│   └── benchmarks/               # Policy benchmarking
│
├── training/
│   ├── demand/                   # Demand model training
│   ├── cost/                     # Cost model training
│   ├── rl/                       # RL policy training
│   ├── anomaly/                  # Anomaly detection training
│   └── pipelines/                # Training pipelines
│
├── inference/
│   ├── decision_orchestrator.py  # Decision orchestration
│   └── realtime_executor.py      # Real-time execution
│
├── data/
│   ├── schemas/                  # Data schemas
│   ├── validators/               # Data validation
│   ├── loaders/                  # Data loading
│   └── feature_store/            # Feature storage
│
├── registry/
│   ├── models/                   # Model registry
│   ├── policies/                 # Policy registry
│   └── metrics/                  # Metrics registry
│
├── evaluation/
│   ├── kpi_definitions.py        # KPI calculations
│   ├── policy_comparison.py       # Policy comparison
│   └── regression_tests.py       # Regression testing
│
├── scheduler/
│   ├── nightly_planning.py       # Nightly planning jobs
│   └── retraining_jobs.py        # Retraining jobs
│
├── config/
│   ├── env.yaml                  # Environment configuration
│   ├── rewards.yaml              # Reward configuration
│   ├── constraints.yaml          # Constraint configuration
│   └── thresholds.yaml           # Threshold configuration
│
└── README.md
```

## Key Components

### 1. Forecast Service
- Probabilistic demand forecasting (P50/P90/P99)
- LightGBM-based MVP with TFT option
- Versioned model registry

### 2. Scenario Service
- Synthetic future generation
- Monte Carlo sampling
- Disruption injection

### 3. Cost Service
- Logistics cost modeling
- Reward shaping integration
- KPI forecasting

### 4. State Service
- Graph-based state encoding
- GNN or engineered features
- Embedding generation

### 5. Policy Service
- Multi-agent RL with PPO/MAPPO
- Centralized critic
- Simulation-only training

### 6. Routing Service
- OR-Tools integration
- RL refinement
- Hybrid optimization

### 7. Safety Service
- Constraint enforcement
- Capacity/SLA/legal rules
- Override mechanisms

### 8. Anomaly Service
- Statistical & ML-based detection
- Multi-type anomaly monitoring
- Re-optimization triggering

### 9. Explain Service
- SHAP-based explainability
- Template-driven explanations
- Confidence scoring

## Core Principles

1. **Backend-only**: No UI code, only ML brain and services
2. **ML-first**: All 9 ML systems must exist (even MVP versions)
3. **No online learning**: All training is offline, inference is read-only
4. **Everything logged**: State → Action → Constraints → Outcome → Metrics
5. **Safety paramount**: Safety model overrides ML decisions
6. **Everything versioned**: Models, rewards, simulations, policies

## Training Pipeline

The system uses a comprehensive offline training approach:
- Simulation engine as core training backbone
- Deterministic, seeded, replayable environments
- Curriculum-based RL training
- Benchmark comparison required

## Deployment

The system is designed for containerized deployment with:
- Read-only inference containers
- Versioned model registries
- Scheduled retraining jobs
- Comprehensive monitoring and alerting

## Getting Started

1. Install dependencies
2. Configure environment in `config/env.yaml`
3. Run training pipelines in `training/pipelines/`
4. Deploy inference services using `inference/decision_orchestrator.py`
5. Schedule jobs using `scheduler/` modules

## Safety & Compliance

- All decisions are logged for audit
- Safety constraints are enforced at multiple levels
- Legal and business rules are codified
- Rollback capabilities for all versions

---
*This is a backend ML system - no frontend/UI components included.*