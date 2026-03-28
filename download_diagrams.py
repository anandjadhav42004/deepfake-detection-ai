import urllib.request
import base64
import json
import os

diagrams = {
    "1_System_Workflow.png": """flowchart TD
    A([Start App Session]) --> B(Access Landing Page)
    B --> C{Select Input Method?}
    C -->|Text Scan| D[Input Text / News Article]
    D --> E[Submit via AJAX]
    E --> F[Flask Backend: /predict_news]
    F --> G[Pre-process & Vectorize Text]
    G --> H[Inference: Scikit-Learn NLP Model]
    H --> I[API Cross-verify: Wikipedia Data]
    I --> J[Calculate linguistic/sensationalism penalties]
    J --> K[Generate Final Tri-state Score]
    C -->|Media Scan| L[Upload Image / Video File]
    L --> M[Submit via AJAX]
    M --> N[Flask Backend: /predict_deepfake]
    N --> O{Is Upload a Video?}
    O -->|Yes| P[OpenCV: Extract 5 equidistant frames]
    O -->|No| Q[PIL: Load & Resize Image]
    P & Q --> R[Pre-process: Normalize Tensor]
    R --> S[Inference: PyTorch DeepfakeCNN]
    S --> T[Aggregate / Average CNN Confidence]
    T --> U[Generate Final Fake Probability]
    K & U --> V(Return JSON Response to Frontend)
    V --> W[Frontend parsing & animation rendering]
    W --> X([Display Verification Dashboard])""",
    
    "2_Sequence_Diagram.png": """sequenceDiagram
    participant U as User
    participant F as Frontend Browser
    participant API as Flask Server routes.py
    participant ML as Local ML Engine
    participant Ext as Wikipedia API
    U->>F: Inputs data Paste text or Drop file
    U->>F: Clicks Initiate Scan
    F->>F: Render loading matrix desktop overlay
    F->>API: HTTP POST Request FormData
    API->>API: Validate Request Payload
    alt Route is predict_deepfake
        API->>ML: Send normalized image tensor to PyTorch
        ML-->>API: Return sigmoid float [0.0 to 1.0]
        API->>API: Map float to Fake/Real confidence
    else Route is predict_news
        API->>ML: Transform string & pass to NLP Pickle
        ML-->>API: Return Fake/Real classification
        API->>Ext: GET request with extracted keywords
        Ext-->>API: Return JSON with article matches
        API->>API: Calculate total multi-factor score
    end
    API-->>F: HTTP 200 OK Verification JSON
    F->>F: Parse scores map to UI metrics
    F->>U: Hide loader & Display Final Dashboard""",
    
    "3_UI_Architecture.png": """graph TD
    A[Browser / Document Object Model] -->|Loads| B[index.html]
    B --> C(Background & Globals)
    C --> C1[.matrix-background]
    C --> C2[.space-background & .grain-overlay]
    B --> D(Navigation Bar)
    D --> D1[Logo & Version]
    D --> D2[System Status Pill]
    B --> E(View Manager)
    E --> F[Landing Page View]
    F --> F1[Hero Section Authenticity Defined]
    F --> F2[Entry Actions: Text or Media Scan]
    F --> F3[Feature Grid]
    E --> G[Application Container]
    G --> G1[Text Scanner Panel]
    G1 --> G1A[Input Textarea]
    G1 --> G1B[Sample Prompt Buttons]
    G --> G2[Media Scanner Panel]
    G2 --> G2A[Drag & Drop Zone]
    G2 --> G2B[Video/Image Preview Window]
    G --> G3[Results Dashboard]
    G3 --> G3A[Score Radial Chart]
    G3 --> G3B[Verdict & Metadata Tags]
    G3 --> G3C[Metrics Breakdown Bars]
    G --> G4[Processing Loader Modal]"""
}

def get_mermaid_url(code):
    state = {"code": code, "mermaid": {"theme": "default"}}
    json_str = json.dumps(state)
    b64_str = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
    b64_str = b64_str.replace("+", "-").replace("/", "_")
    return f"https://mermaid.ink/img/{b64_str}"

output_dir = "/Users/anand/Desktop/deepfake-detection-ai"
for filename, code in diagrams.items():
    print(f"Downloading {filename}...")
    url = get_mermaid_url(code)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        out_path = os.path.join(output_dir, filename)
        with urllib.request.urlopen(req) as response, open(out_path, 'wb') as f:
            f.write(response.read())
        print(f"✅ Saved {filename} successfully to your project folder!")
    except Exception as e:
        print(f"❌ Failed to fetch {filename}: {e}")
