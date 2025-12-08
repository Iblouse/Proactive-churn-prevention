```graph TD
    %% -- STYLING DEFINITIONS --
    %% Main Container
    classDef container fill:#ffffff,stroke:#333333,stroke-width:2px,color:#000000;
    
    %% Phase Colors (Soft Pastels for readability)
    classDef p1 fill:#e3f2fd,stroke:#1565c0,stroke-width:1px,color:#000; %% Blue: Data
    classDef p2 fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px,color:#000; %% Green: Behavioral
    classDef p3 fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,color:#000; %% Purple: Predictive
    classDef p4 fill:#fff3e0,stroke:#ef6c00,stroke-width:1px,color:#000; %% Orange: Intervention
    classDef p5 fill:#ffebee,stroke:#c62828,stroke-width:1px,color:#000; %% Red: Evaluation
    classDef p6 fill:#eceff1,stroke:#455a64,stroke-width:1px,color:#000; %% Grey: Reporting

    %% Main Container
    subgraph Main_System [Proactive Churn Prevention System]
        direction TB

        %% --- Phase 1: Data Input (Blue) ---
        Start([Start: Customer<br/>Data Ingestion]) --> P1[Feature Engineering<br/>& Segmentation]
        P1 -->|Segmented Data| P2

        %% --- Phase 2: Behavioral Monitoring (Green) ---
        P2[Monitor Engagement<br/>Signals] --> D1{Risk<br/>Detected?}
        D1 -- No: Continued Monitoring --> P2
        D1 -- Yes: Drop > Threshold<br/>OR Payment Issue --> P3

        %% --- Phase 3: Risk Prediction (Purple) ---
        P3[Calculate Churn<br/>Probability] --> P4[Classify Risk Tier]
        P4 --> D2

        %% --- Phase 4: Intervention Selection (Orange) ---
        D2{Is there<br/>Risk?}
        D2 -- Yes --> P6[Execute<br/>Intervention]
        P6 --> P7[Assign A/B<br/>Test Variant]
        P7 --> P8
        
        %% "No" branch skips to next phase
        D2 -- No --> P8

        %% --- Phase 5 & 6: Evaluation (Red) ---
        P8[Track Outcomes<br/>60-Day Window] --> D4{Customer<br/>Churned?}
        D4 -- No --> R1[Record SUCCESS<br/>& Calc ROI]
        D4 -- Yes --> R2[Record FAILURE<br/>& Flag]
        
        R1 & R2 --> P9[Update Model &<br/>Statistical Significance]

        %% --- Phase 7 & 8: Reporting & Decision (Grey) ---
        P9 -.->|Feedback Loop| P3
        P9 --> P10[Generate Executive<br/>Dashboard]
        P10 --> D5{Stakeholder<br/>Approval?}

        D5 -- No: Refine --> P1
        D5 -- Yes --> End([Deploy to Production<br/>& Scale])
    end

    %% -- APPLYING CLASSES --
    class Main_System container;
    
    %% Phase 1
    class Start,P1 p1;
    %% Phase 2
    class P2,D1 p2;
    %% Phase 3
    class P3,P4 p3;
    %% Phase 4
    class D2,P6,P7 p4;
    %% Phase 5 & 6
    class P8,D4,R1,R2,P9 p5;
    %% Phase 7 & 8
    class P10,D5,End p6;
    ```