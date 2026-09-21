import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask,
    request,
    jsonify,
    render_template_string,
)

from openai import OpenAI


# ============================================================
# NEXUS-AI
# Biotechnology Research Copilot
# ============================================================

load_dotenv()

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "nexus_ai.db"

AI_PROVIDER = os.getenv("AI_PROVIDER", "groq").lower()
AI_MODEL = os.getenv("AI_MODEL", "llama-3.3-70b-versatile")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


# ============================================================
# DATABASE
# ============================================================

def init_db():
    conn = sqlite3.connect(DB_FILE)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            agents TEXT,
            answer TEXT,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ============================================================
# AI CLIENT
# ============================================================

def get_client():
    if AI_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add it to your .env file."
            )

        return OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )

    if AI_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Add it to your .env file."
            )

        return OpenAI(api_key=OPENAI_API_KEY)

    raise RuntimeError(
        f"Unsupported AI_PROVIDER: {AI_PROVIDER}"
    )


# ============================================================
# AGENTS
# ============================================================

AGENTS = {
    "ALICE": {
        "name": "ALICE",
        "title": "Research & Literature",
        "icon": "🧬",
        "prompt": """
You are ALICE, the Research & Literature specialist of NEXUS-AI.

Your responsibilities:
- Explain scientific concepts.
- Organize research questions.
- Suggest literature-review directions.
- Identify important concepts, mechanisms and terminology.
- Clearly distinguish established knowledge from hypotheses.
- Never invent papers, citations, experiments or results.

Answer at a university/research level but explain difficult concepts clearly.
"""
    },

    "GENE": {
        "name": "GENE",
        "title": "Biology & Genetics",
        "icon": "🧪",
        "prompt": """
You are GENE, the Biology and Genetics specialist of NEXUS-AI.

Focus on:
- Molecular biology
- Genetics
- Genomics
- Cell biology
- Biotechnology
- Microbiology
- CRISPR concepts
- Proteins
- DNA/RNA
- Biological mechanisms

Explain mechanisms step-by-step when appropriate.

Do not fabricate biological facts or experimental results.
For potentially dangerous biological procedures, remain safety-aware
and provide educational high-level information rather than actionable
harmful protocols.
"""
    },

    "DOC": {
        "name": "DOC",
        "title": "Scientific Documentation",
        "icon": "📄",
        "prompt": """
You are DOC, the scientific documentation specialist.

Convert research ideas into:
- Abstracts
- Research proposals
- Literature reviews
- Reports
- PPT structures
- Scientific summaries
- Project documentation

Use professional scientific writing.
Keep claims evidence-aware.
Do not invent references.
"""
    },

    "DATA": {
        "name": "DATA",
        "title": "Data Analysis",
        "icon": "📊",
        "prompt": """
You are DATA, the scientific data-analysis specialist.

Focus on:
- Dataset interpretation
- Descriptive statistics
- Trends
- Relationships
- Experimental data interpretation
- Tables
- Graph suggestions
- Scientific reasoning

Never fabricate numerical results.
If data is unavailable, explicitly say what information is needed.
"""
    },

    "NOVA": {
        "name": "NOVA",
        "title": "Biotech Innovation",
        "icon": "🚀",
        "prompt": """
You are NOVA, the biotechnology innovation specialist.

Generate innovative but scientifically grounded ideas involving:
- New medicines
- Diagnostics
- Bioinformatics
- Synthetic biology
- Environmental biotechnology
- Agricultural biotechnology
- Personalized medicine
- AI + biotechnology
- Research tools
- Future healthcare

For each idea consider:
Problem
Solution
Biological principle
Technology
Target users
Innovation
Research requirements
Challenges
Future scope

Avoid presenting speculative ideas as established science.
"""
    }
}


# ============================================================
# AI CALL
# ============================================================

def ask_ai(system_prompt, user_prompt):
    client = get_client()

    response = client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": user_prompt.strip()
            }
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content


# ============================================================
# AGENT SELECTION
# ============================================================

def auto_select_agents(question):
    q = question.lower()

    selected = set()

    research_words = [
        "research", "paper", "literature",
        "study", "review", "evidence",
        "scientific", "article"
    ]

    biology_words = [
        "dna", "rna", "gene", "genetic",
        "protein", "cell", "crispr",
        "bacteria", "virus", "biology",
        "genetics", "microbiology"
    ]

    data_words = [
        "data", "dataset", "csv",
        "statistics", "analysis",
        "mean", "correlation",
        "graph", "percentage"
    ]

    document_words = [
        "report", "ppt", "presentation",
        "abstract", "proposal",
        "assignment", "documentation"
    ]

    innovation_words = [
        "idea", "innovation", "startup",
        "medicine", "drug", "future",
        "invention", "diagnostic",
        "product"
    ]

    if any(word in q for word in research_words):
        selected.add("ALICE")

    if any(word in q for word in biology_words):
        selected.add("GENE")

    if any(word in q for word in data_words):
        selected.add("DATA")

    if any(word in q for word in document_words):
        selected.add("DOC")

    if any(word in q for word in innovation_words):
        selected.add("NOVA")

    if not selected:
        selected = {"ALICE", "GENE"}

    return list(selected)


# ============================================================
# AGENT EXECUTION
# ============================================================

def run_agent(agent_id, question):
    agent = AGENTS[agent_id]

    prompt = f"""
User research question:

{question}

Provide your specialist analysis.

Your role:
{agent["title"]}

Keep the answer factual, structured and useful.
"""

    return ask_ai(agent["prompt"], prompt)


# ============================================================
# ORCHESTRATOR
# ============================================================

def run_nexus(question, selected_agents=None):

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not selected_agents:
        selected_agents = auto_select_agents(question)

    results = {}

    for agent_id in selected_agents:
        if agent_id not in AGENTS:
            continue

        try:
            results[agent_id] = run_agent(
                agent_id,
                question
            )
        except Exception as e:
            results[agent_id] = (
                f"Agent error: {type(e).__name__}: {str(e)}"
            )

    combined = "\n\n".join(
        f"### {agent_id} — {AGENTS[agent_id]['title']}\n"
        f"{answer}"
        for agent_id, answer in results.items()
    )

    synthesis_prompt = f"""
You are the NEXUS-AI ORCHESTRATOR.

You received specialist analyses from multiple biotechnology agents.

Original question:
{question}

Specialist analyses:

{combined}

Create one final answer.

Requirements:
1. Synthesize the strongest useful information.
2. Do not blindly repeat conflicting claims.
3. Clearly distinguish established facts from hypotheses.
4. Do not invent citations or research results.
5. Use headings and bullet points when useful.
6. If the question is scientific, explain difficult concepts clearly.
7. Mention uncertainty where appropriate.
"""

    try:
        final_answer = ask_ai(
            synthesis_prompt,
            "Synthesize the specialist analyses into the final NEXUS-AI response."
        )
    except Exception as e:
        final_answer = combined + (
            f"\n\nORCHESTRATOR ERROR: {type(e).__name__}: {str(e)}"
        )

    conn = sqlite3.connect(DB_FILE)

    conn.execute(
        """
        INSERT INTO history
        (question, agents, answer, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            question,
            json.dumps(selected_agents),
            final_answer,
            datetime.now().isoformat(timespec="seconds")
        )
    )

    conn.commit()
    conn.close()

    return {
        "question": question,
        "agents": selected_agents,
        "specialists": results,
        "answer": final_answer
    }


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/api/agents")
def agents():
    return jsonify(AGENTS)


@app.route("/api/status")
def status():
    return jsonify({
        "status": "online",
        "provider": AI_PROVIDER,
        "model": AI_MODEL,
        "agents": list(AGENTS.keys()),
        "api_configured": bool(
            GROQ_API_KEY if AI_PROVIDER == "groq"
            else OPENAI_API_KEY
        )
    })


@app.route("/api/research", methods=["POST"])
def research():

    data = request.get_json(silent=True) or {}

    question = data.get("question", "").strip()
    selected_agents = data.get("agents")

    if not question:
        return jsonify({
            "error": "Please enter a research question."
        }), 400

    try:
        result = run_nexus(
            question,
            selected_agents
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": f"{type(e).__name__}: {str(e)}"
        }), 500


@app.route("/api/history")
def history():

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT id, question, agents,
               answer, created_at
        FROM history
        ORDER BY id DESC
        LIMIT 50
        """
    ).fetchall()

    conn.close()

    return jsonify([
        dict(row)
        for row in rows
    ])


@app.route("/api/innovation", methods=["POST"])
def innovation():

    data = request.get_json(silent=True) or {}

    idea = data.get("idea", "").strip()

    if not idea:
        return jsonify({
            "error": "Enter an innovation idea."
        }), 400

    prompt = f"""
Develop this biotechnology innovation idea:

{idea}

Return:

1. Problem
2. Proposed Solution
3. Biological Principle
4. Technology
5. Target Users
6. Innovation
7. Research Requirements
8. Challenges
9. Ethical Considerations
10. Future Scope

Clearly label speculative concepts.
"""

    try:
        result = ask_ai(
            AGENTS["NOVA"]["prompt"],
            prompt
        )

        return jsonify({
            "idea": idea,
            "result": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/api/analyze-csv", methods=["POST"])
def analyze_csv():

    if "file" not in request.files:
        return jsonify({
            "error": "No CSV file uploaded."
        }), 400

    file = request.files["file"]

    if not file.filename.lower().endswith(".csv"):
        return jsonify({
            "error": "Only CSV files are supported."
        }), 400

    try:
        df = pd.read_csv(file)

        summary = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "missing_values": {
                str(k): int(v)
                for k, v in df.isna().sum().items()
            },
            "statistics": json.loads(
                df.describe(
                    include="all"
                ).fillna("").to_json()
            )
        }

        return jsonify(summary)

    except Exception as e:
        return jsonify({
            "error": f"CSV analysis failed: {str(e)}"
        }), 500


# ============================================================
# FRONTEND
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>NEXUS-AI</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Inter, Arial, sans-serif;
    background:
        radial-gradient(circle at top right,
        #102a43 0,
        #050b14 35%,
        #02050a 75%);
    color: #e8f3ff;
    min-height: 100vh;
}

.container {
    display: flex;
    min-height: 100vh;
}

.sidebar {
    width: 245px;
    padding: 25px 18px;
    background: rgba(5, 12, 22, .82);
    border-right: 1px solid rgba(255,255,255,.08);
    backdrop-filter: blur(20px);
}

.logo {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 5px;
}

.tagline {
    font-size: 11px;
    color: #7f9bb5;
    margin-bottom: 35px;
}

.nav button {
    width: 100%;
    border: 0;
    background: transparent;
    color: #9db2c8;
    padding: 13px;
    margin: 4px 0;
    text-align: left;
    border-radius: 10px;
    cursor: pointer;
}

.nav button:hover,
.nav button.active {
    background: rgba(70,170,255,.12);
    color: #ffffff;
}

.main {
    flex: 1;
    padding: 35px;
    overflow-y: auto;
}

.header h1 {
    margin: 0;
    font-size: 38px;
}

.header p {
    color: #8ca4bb;
}

.panel {
    margin-top: 25px;
    background: rgba(8,18,31,.72);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 18px;
    padding: 22px;
    backdrop-filter: blur(20px);
}

textarea {
    width: 100%;
    min-height: 150px;
    resize: vertical;
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 12px;
    background: #050b14;
    color: white;
    padding: 17px;
    font-size: 15px;
    outline: none;
}

textarea:focus {
    border-color: #319cff;
}

.agents {
    display: grid;
    grid-template-columns:
        repeat(auto-fit,minmax(170px,1fr));
    gap: 12px;
    margin-top: 18px;
}

.agent {
    border: 1px solid rgba(255,255,255,.08);
    background: rgba(255,255,255,.025);
    border-radius: 14px;
    padding: 15px;
    cursor: pointer;
}

.agent.selected {
    border-color: #319cff;
    background: rgba(49,156,255,.1);
}

.agent-icon {
    font-size: 25px;
}

.agent-name {
    font-weight: 700;
    margin-top: 8px;
}

.agent-title {
    color: #8098ae;
    font-size: 12px;
    margin-top: 4px;
}

.actions {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 18px;
}

button.primary,
button.secondary {
    border: 0;
    padding: 13px 20px;
    border-radius: 10px;
    cursor: pointer;
    font-weight: 700;
}

button.primary {
    background: linear-gradient(135deg,#168cff,#6b5cff);
    color: white;
}

button.secondary {
    background: #101d2d;
    color: #bcd0e5;
}

.result {
    white-space: pre-wrap;
    line-height: 1.7;
    color: #dcecff;
}

.status {
    margin-top: 15px;
    color: #72b9ff;
}

.history-item {
    padding: 15px;
    border-bottom: 1px solid rgba(255,255,255,.06);
}

.history-question {
    font-weight: 700;
}

.small {
    font-size: 12px;
    color: #71879b;
}

pre {
    white-space: pre-wrap;
    overflow-x: auto;
}

@media(max-width:800px) {

    .container {
        flex-direction: column;
    }

    .sidebar {
        width: 100%;
        border-right: 0;
        border-bottom: 1px solid rgba(255,255,255,.08);
    }

    .main {
        padding: 20px;
    }

    .header h1 {
        font-size: 28px;
    }

}

</style>

</head>

<body>

<div class="container">

<aside class="sidebar">

<div class="logo">
NEXUS-AI
</div>

<div class="tagline">
Connect Intelligence. Accelerate Discovery.
</div>

<div class="nav">

<button onclick="showPage('research')" class="active">
🧬 Research
</button>

<button onclick="showPage('agents')">
🤖 Agents
</button>

<button onclick="showPage('history')">
🕘 History
</button>

<button onclick="showPage('innovation')">
🚀 Innovation Lab
</button>

<button onclick="showPage('data')">
📊 Data Analysis
</button>

<button onclick="showPage('settings')">
⚙ Settings
</button>

</div>

</aside>


<main class="main">

<div id="researchPage">

<div class="header">

<h1>Biotechnology Research Copilot</h1>

<p>
Ask a scientific question and let the NEXUS agents work together.
</p>

</div>


<div class="panel">

<textarea
id="question"
placeholder="Example: Explain how CRISPR-Cas9 works and suggest research directions..."
></textarea>


<div id="agents"
     class="agents"></div>


<div class="actions">

<button class="primary"
        onclick="runResearch()">
RUN NEXUS
</button>

<button class="secondary"
        onclick="autoAgents()">
AUTO SELECT
</button>

<button class="secondary"
        onclick="document.getElementById('question').value=''">
CLEAR
</button>

</div>


<div id="status"
     class="status"></div>

</div>


<div class="panel">

<h2>Research Output</h2>

<div id="result"
     class="result">
Your synthesized research answer will appear here.
</div>

</div>

</div>


<div id="agentsPage"
     style="display:none">

<h1>AI Agents</h1>

<div id="agentInfo"></div>

</div>


<div id="historyPage"
     style="display:none">

<h1>Research History</h1>

<div class="panel"
     id="history">
Loading...
</div>

</div>


<div id="innovationPage"
     style="display:none">

<h1>Innovation Lab</h1>

<div class="panel">

<textarea
id="innovationIdea"
placeholder="Example: AI-assisted personalized cancer diagnostics"
></textarea>

<div class="actions">

<button class="primary"
        onclick="runInnovation()">
GENERATE INNOVATION
</button>

</div>

</div>

<div class="panel">

<div id="innovationResult">
NOVA output will appear here.
</div>

</div>

</div>


<div id="dataPage"
     style="display:none">

<h1>Scientific Data Analysis</h1>

<div class="panel">

<input
type="file"
id="csvFile"
accept=".csv">

<div class="actions">

<button class="primary"
        onclick="analyzeCSV()">
ANALYZE CSV
</button>

</div>

<pre id="dataResult"></pre>

</div>

</div>


<div id="settingsPage"
     style="display:none">

<h1>Settings</h1>

<div class="panel"
     id="settings">
Checking system...
</div>

</div>

</main>

</div>


<script>

let selectedAgents = [];

async function loadAgents() {

    const response =
        await fetch("/api/agents");

    const data =
        await response.json();

    const box =
        document.getElementById("agents");

    box.innerHTML = "";

    Object.values(data).forEach(agent => {

        const div =
            document.createElement("div");

        div.className = "agent";

        div.innerHTML = `
            <div class="agent-icon">
                ${agent.icon}
            </div>

            <div class="agent-name">
                ${agent.name}
            </div>

            <div class="agent-title">
                ${agent.title}
            </div>
        `;

        div.onclick = () => {

            if(selectedAgents.includes(agent.name)) {

                selectedAgents =
                    selectedAgents.filter(
                        x => x !== agent.name
                    );

                div.classList.remove("selected");

            } else {

                selectedAgents.push(agent.name);

                div.classList.add("selected");
            }
        };

        box.appendChild(div);

    });
}


async function runResearch() {

    const question =
        document.getElementById("question").value.trim();

    if(!question) {

        alert("Enter a research question first.");

        return;
    }

    const status =
        document.getElementById("status");

    const result =
        document.getElementById("result");

    status.innerText =
        "NEXUS orchestrator is activating agents...";

    result.innerText =
        "Working...";

    try {

        const response =
            await fetch("/api/research", {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    question: question,
                    agents:
                        selectedAgents.length
                        ? selectedAgents
                        : null
                })

            });

        const data =
            await response.json();

        if(data.error) {

            result.innerText =
                data.error;

            status.innerText =
                "System error.";

            return;
        }

        result.innerText =
            data.answer;

        status.innerText =
            "Completed ✓ Agents: "
            + data.agents.join(", ");

    } catch(error) {

        status.innerText =
            "Connection error.";

        result.innerText =
            error.toString();
    }
}


async function autoAgents() {

    const question =
        document.getElementById("question").value.trim();

    if(!question) {

        alert("Enter a question first.");

        return;
    }

    const response =
        await fetch("/api/research", {

            method: "POST",

            headers: {
                "Content-Type":
                    "application/json"
            },

            body: JSON.stringify({
                question: question
            })

        });

    const data =
        await response.json();

    if(data.agents) {

        selectedAgents =
            data.agents;

        document
            .querySelectorAll(".agent")
            .forEach(card => {

                const name =
                    card.querySelector(
                        ".agent-name"
                    ).innerText;

                card.classList.toggle(
                    "selected",
                    selectedAgents.includes(name)
                );

            });
    }
}


async function loadHistory() {

    const response =
        await fetch("/api/history");

    const data =
        await response.json();

    const box =
        document.getElementById("history");

    if(!data.length) {

        box.innerHTML =
            "No research history yet.";

        return;
    }

    box.innerHTML =
        data.map(item => `

        <div class="history-item">

            <div class="history-question">
                ${escapeHtml(item.question)}
            </div>

            <div class="small">
                ${item.created_at}
            </div>

            <p>
                ${escapeHtml(item.answer)}
            </p>

        </div>

    `).join("");
}


async function runInnovation() {

    const idea =
        document
        .getElementById("innovationIdea")
        .value.trim();

    if(!idea) {

        alert("Enter an idea.");

        return;
    }

    const box =
        document.getElementById(
            "innovationResult"
        );

    box.innerText =
        "NOVA is generating innovation...";

    const response =
        await fetch("/api/innovation", {

            method: "POST",

            headers: {
                "Content-Type":
                    "application/json"
            },

            body: JSON.stringify({
                idea: idea
            })

        });

    const data =
        await response.json();

    box.innerText =
        data.result || data.error;
}


async function analyzeCSV() {

    const file =
        document
        .getElementById("csvFile")
        .files[0];

    if(!file) {

        alert("Select a CSV file.");

        return;
    }

    const form =
        new FormData();

    form.append("file", file);

    const response =
        await fetch(
            "/api/analyze-csv",
            {
                method: "POST",
                body: form
            }
        );

    const data =
        await response.json();

    document
        .getElementById("dataResult")
        .innerText =
        JSON.stringify(
            data,
            null,
            2
        );
}


async function loadSettings() {

    const response =
        await fetch("/api/status");

    const data =
        await response.json();

    document
        .getElementById("settings")
        .innerHTML = `

        <p>
        <b>Status:</b>
        ${data.status}
        </p>

        <p>
        <b>Provider:</b>
        ${data.provider}
        </p>

        <p>
        <b>Model:</b>
        ${data.model}
        </p>

        <p>
        <b>API:</b>
        ${data.api_configured
            ? "Configured ✓"
            : "Missing ✕"}
        </p>

        `;
}


function showPage(page) {

    const pages = [
        "research",
        "agents",
        "history",
        "innovation",
        "data",
        "settings"
    ];

    pages.forEach(name => {

        const element =
            document.getElementById(
                name + "Page"
            );

        if(element)
            element.style.display =
                name === page
                ? "block"
                : "none";
    });

    if(page === "history")
        loadHistory();

    if(page === "settings")
        loadSettings();

    if(page === "agents")
        renderAgentInfo();

}


function renderAgentInfo() {

    fetch("/api/agents")
    .then(r => r.json())
    .then(data => {

        document
        .getElementById("agentInfo")
        .innerHTML =
            Object.values(data)
            .map(agent => `

                <div class="panel">

                    <h2>
                        ${agent.icon}
                        ${agent.name}
                    </h2>

                    <p>
                        ${agent.title}
                    </p>

                </div>

            `)
            .join("");

    });
}


function escapeHtml(text) {

    const div =
        document.createElement("div");

    div.innerText =
        text || "";

    return div.innerHTML;
}


loadAgents();

</script>

</body>

</html>
"""


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("              NEXUS-AI")
    print("     Biotechnology Research Copilot")
    print("=" * 60)
    print()
    print(f"Provider : {AI_PROVIDER}")
    print(f"Model    : {AI_MODEL}")
    print(f"Database : {DB_FILE}")
    print()
    print("Open:")
    print("http://127.0.0.1:5000")
    print()
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=int(
            os.getenv("PORT", "5000")
        ),
        debug=False
    )
