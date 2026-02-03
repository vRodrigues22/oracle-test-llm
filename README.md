Oracle Test — LLM Semantic Test Oracle


📘 Visão Geral

O Oracle Test é um framework para avaliação automática de saídas de sistemas usando:

- Oráculo Tradicional (regex / similaridade)
- Oráculo Semântico baseado em LLM

Suporta múltiplos provedores:

- Google Gemini
- OpenAI
- LLaMA (via Ollama ou endpoint OpenAI-compatible)


🏗 Estrutura do Projeto:

oracle_test/

│

├── requirements.txt

├── .env.example

├── README.md

│

├── config.py

├── cache.py

├── costs.py

├── schemas.py

│

├── llm_oracle.py

├── traditional_oracle.py

├── experiment_runner.py

├── metrics.py

│

├── prompts/

│   ├── llm_system.txt

│   └── llm_instructions.txt

│

├── data/

│   ├── test_cases.json

│   └── ground_truth.json

│

└── results/


⚙️ Requisitos

Python 3.10+
Windows / Linux / MacOS
Conta em pelo menos um provedor LLM

__________________________________________________________

🚀 Instalação

1) Criar ambiente virtual:

- Windows:

python -m venv .venv

Entrar na venv:
.venv\Scripts\activate


- Linux / Mac:

python3 -m venv .venv

Entrar na venv:
source .venv/bin/activate


2) Instalar dependências:

pip install -r requirements.txt

__________________________________________________________

🔐 Configuração .env:

1) Exemplo .env — Gemini:

LLM_PROVIDER=gemini
LLM_MODEL=gemini-3-flash-preview
GEMINI_API_KEY=COLE_SUA_CHAVE

LLM_ENABLE_CACHE=false
RUN_ID=RUN-001
AUTO_GENERATE_TESTS=false


2) Exemplo .env — OpenAI:

LLM_PROVIDER=openai
LLM_MODEL=gpt-5.2
OPENAI_API_KEY=COLE_SUA_CHAVE

⚠ ChatGPT Plus NÃO inclui créditos da API OpenAI.


3) Exemplo .env — LLaMA (Ollama):

LLM_PROVIDER=llama
LLM_MODEL=llama3
LLAMA_BASE_URL=http://localhost:11434/v1

__________________________________________________________

📊 Rodar o experimento completo:

1) Verifique se data/ tem os arquivos

Rode:
dir data


2) Execute o experimento (com --smoke)

Rode:
python experiment_runner.py --smoke


3) Execute o experimento completo (sem --smoke)

Rode:
python experiment_runner.py

Isso deve gerar saídas na pasta results/ (ex.: .jsonl e .md, dependendo do seu runner).


Depois confira:
dir results

4) Execute para validar o relatório:

Rode:
type results\report_DATASET-001.md

