# Proactive Churn Prevention System - Workflow Diagrams

## Notebook Execution Flow

This flowchart matches the actual notebook execution order (Sections 1-13).

```mermaid
graph TD
    %% -- STYLING DEFINITIONS --
    classDef container fill:#ffffff,stroke:#333333,stroke-width:2px,color:#000000;
    classDef p1 fill:#e3f2fd,stroke:#1565c0,stroke-width:1px,color:#000;
    classDef p2 fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px,color:#000;
    classDef p3 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,color:#000;
    classDef p4 fill:#fff3e0,stroke:#ef6c00,stroke-width:1px,color:#000;
    classDef p5 fill:#ffebee,stroke:#c62828,stroke-width:1px,color:#000;
    classDef p6 fill:#eceff1,stroke:#455a64,stroke-width:1px,color:#000;

    %% Main Container
    subgraph Main_System [Proactive Churn Prevention System - Notebook Flow]
        direction TB

        %% --- Phase 1: Data Foundation (Blue) - Sections 1-2 ---
        Start([Section 1-2:<br/>Setup & Data]) --> P1[Generate Synthetic<br/>Customer Data<br/>n=3,000]
        P1 --> P2[Feature Engineering<br/>15 Features + CLV]
        P2 -->|customer_df| P3

        %% --- Phase 2: Modeling (Green) - Section 3 ---
        P3[Train Churn Model<br/>LogisticRegression] --> P4[Add churn_probability<br/>& risk_tier]
        P4 --> P5[Train Survival Model<br/>Cox PH]
        P5 --> P6[Calculate Optimal<br/>Intervention Window<br/>Day 46-95]
        P6 --> P7[Feature Importance<br/>& Actionability]

        %% --- Phase 3: Evaluation (Purple) - Section 4 ---
        P7 --> P8[Model Evaluation<br/>AUC: 0.66, C-Index: 0.68]
        P8 --> D1{Metrics<br/>Adequate?}
        D1 -- No: Retrain --> P3
        D1 -- Yes --> P9

        %% --- Phase 4: Agent Construction (Orange) - Sections 5-6 ---
        P9[Define Tool Functions<br/>recommend_intervention] --> P10[Create AI Agents<br/>Orchestrator + 4 Specialists]
        P10 --> P11[Configure Infrastructure<br/>Memory & Observability]

        %% --- Phase 5: Validation (Red) - Sections 9-10 ---
        P11 --> P12[Local Testing<br/>Validate Tools]
        P12 --> D2{Tools<br/>Working?}
        D2 -- No: Debug --> P9
        D2 -- Yes --> P13

        P13[A/B Test Simulation<br/>5 Variants × 400 each] --> P14[Statistical Analysis<br/>Chi-Square + Bonferroni]
        P14 --> P15[Calculate ROI<br/>per Channel]
        P15 --> D3{Significant<br/>Results?}
        D3 -- No: Adjust --> P13
        D3 -- Yes --> P16

        %% --- Phase 6: Output (Grey) - Sections 11-13 ---
        P16[Generate Executive<br/>Dashboard<br/>6 Visualizations] --> D4{Stakeholder<br/>Approval?}
        D4 -- No: Refine --> P1
        D4 -- Yes --> P17[Package for<br/>Deployment]
        P17 --> End([Section 13:<br/>Cleanup & Deploy])
    end

    %% -- APPLYING CLASSES --
    class Main_System container;
    class Start,P1,P2 p1;
    class P3,P4,P5,P6,P7 p2;
    class P8,D1 p3;
    class P9,P10,P11 p4;
    class P12,D2,P13,P14,P15,D3 p5;
    class P16,D4,P17,End p6;
```

## Key Data Dependencies

```mermaid
graph LR
    %% Styling
    classDef data fill:#e3f2fd,stroke:#1565c0,stroke-width:1px,color:#000;
    classDef model fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px,color:#000;
    classDef output fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,color:#000;
    classDef test fill:#ffebee,stroke:#c62828,stroke-width:1px,color:#000;

    subgraph Section2 [Section 2: Data]
        CD[(customer_df<br/>3,000 rows)]
    end

    subgraph Section3 [Section 3: Modeling]
        CM[CHURN_MODEL]
        SM[SURVIVAL_MODEL]
        STATS[SURVIVAL_INTERVENTION<br/>_STATS]
    end

    subgraph Section5 [Section 5: Tools]
        RI[recommend_intervention]
    end

    subgraph Section10 [Section 10: A/B Testing]
        CE[CHANNEL_EFFECTIVENESS]
        ROI[INTERVENTION_ROI]
    end

    subgraph Section11 [Section 11: Dashboard]
        DASH[Executive<br/>Dashboard]
    end

    CD --> CM
    CD --> SM
    SM --> STATS
    STATS --> RI
    CE --> RI
    CM --> RI
    CE --> DASH
    ROI --> DASH
    STATS --> DASH
    CD --> DASH

    class CD data;
    class CM,SM,STATS model;
    class RI output;
    class CE,ROI test;
    class DASH output;
```

## Production System Flow (Conceptual)

This shows how the trained system would operate in production (not in the notebook).

```mermaid
graph TD
    %% -- STYLING DEFINITIONS --
    classDef container fill:#ffffff,stroke:#333333,stroke-width:2px,color:#000000;
    classDef p1 fill:#e3f2fd,stroke:#1565c0,stroke-width:1px,color:#000;
    classDef p2 fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px,color:#000;
    classDef p3 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,color:#000;
    classDef p4 fill:#fff3e0,stroke:#ef6c00,stroke-width:1px,color:#000;
    classDef p5 fill:#ffebee,stroke:#c62828,stroke-width:1px,color:#000;
    classDef p6 fill:#eceff1,stroke:#455a64,stroke-width:1px,color:#000;

    subgraph Production [Production Churn Prevention System]
        direction TB

        %% --- Real-time Monitoring ---
        P1([Customer<br/>Activity]) --> P2[Feature<br/>Extraction]
        P2 --> P3[Churn Score<br/>Prediction]
        P3 --> P4[Risk Tier<br/>Classification]

        %% --- Decision Point ---
        P4 --> D1{High/Critical<br/>Risk?}
        D1 -- No --> P5[Continue<br/>Monitoring]
        P5 -.-> P1

        %% --- Intervention ---
        D1 -- Yes --> P6[Check Timing<br/>Day 46-95?]
        P6 --> D2{In Optimal<br/>Window?}
        D2 -- No: Too Early --> P5
        D2 -- Yes --> P7[Select Channel<br/>Based on ROI]
        P7 --> P8[Execute<br/>Intervention]

        %% --- Outcome Tracking ---
        P8 --> P9[Track 120-Day<br/>Outcome]
        P9 --> D3{Churned?}
        D3 -- No --> R1[SUCCESS<br/>Calculate ROI]
        D3 -- Yes --> R2[FAILURE<br/>Analyze Why]

        %% --- Feedback Loop ---
        R1 & R2 --> P10[Update Channel<br/>Effectiveness]
        P10 -.->|Model Retrain| P3
        P10 --> P11[Dashboard<br/>Reporting]
    end

    %% -- APPLYING CLASSES --
    class Production container;
    class P1,P2 p1;
    class P3,P4,D1 p2;
    class P5 p1;
    class P6,D2,P7,P8 p4;
    class P9,D3,R1,R2,P10 p5;
    class P11 p6;
```

## A/B Testing Analysis Flow

```mermaid
graph LR
    %% Styling
    classDef control fill:#eceff1,stroke:#455a64,stroke-width:1px,color:#000;
    classDef variant fill:#fff3e0,stroke:#ef6c00,stroke-width:1px,color:#000;
    classDef winner fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000;
    classDef stats fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,color:#000;

    subgraph Experiment [Multi-Variant A/B Test]
        C[Control<br/>n=400<br/>19.2% churn]
        E[Email<br/>n=400<br/>15.8% churn]
        D[Discount<br/>n=400<br/>13.2% churn]
        CA[Call<br/>n=400<br/>9.0% churn]
        CO[Combined<br/>n=400<br/>13.5% churn]
    end

    subgraph Analysis [Statistical Analysis]
        CHI[Chi-Square<br/>Test]
        BON[Bonferroni<br/>α = 0.0125]
    end

    subgraph Results [Results]
        W[Winner: Call<br/>53.2% lift<br/>p < 0.0001]
        ROI[ROI Rankings:<br/>Email: 127.9x<br/>Discount: 11.3x<br/>Call: 5.5x]
    end

    C & E & D & CA & CO --> CHI
    CHI --> BON
    BON --> W
    W --> ROI

    class C control;
    class E,D,CO variant;
    class CA winner;
    class CHI,BON stats;
    class W winner;
```

## Section Dependencies

```mermaid
graph TD
    S1[Section 1-2<br/>Setup & Data] --> S3[Section 3<br/>Modeling]
    S3 --> S4[Section 4<br/>Evaluation]
    S4 --> S5[Section 5-6<br/>Tools & Agents]
    S5 --> S7[Section 7-8<br/>Infrastructure]
    S7 --> S9[Section 9<br/>Local Testing]
    S9 --> S10[Section 10<br/>A/B Testing]
    S10 --> S11[Section 11<br/>Dashboard]
    S11 --> S12[Section 12-13<br/>Deploy & Cleanup]

    %% Data dependencies
    S3 -.->|SURVIVAL_STATS| S5
    S10 -.->|CHANNEL_EFFECTIVENESS| S5
    S10 -.->|INTERVENTION_ROI| S11

    style S10 fill:#ffebee,stroke:#c62828
    style S5 fill:#fff3e0,stroke:#ef6c00
```
