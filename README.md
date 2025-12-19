# RAG Vector V2

Aplicação RAG (Retrieval-Augmented Generation) com interface em **Streamlit**, armazenamento vetorial em **PostgreSQL + pgvector** e geração de respostas via **OpenAI**.

A aplicação permite:

- Fazer upload de documentos (`.pdf`, `.docx`, `.txt`, `.md`) ou `.zip` contendo esses arquivos
- Quebrar o conteúdo em *chunks* e gerar *embeddings*
- Persistir os *chunks* e vetores no Postgres (pgvector)
- Consultar por **busca vetorial**, **busca semântica (full-text)** ou **busca híbrida**
- Responder perguntas usando o contexto recuperado (RAG)

---

## 1) Requisitos

### 1.1 Software

- Python 3.10+ (recomendado)
- Docker + Docker Compose (para subir o Postgres/pgvector)

### 1.2 Contas/credenciais

- Chave de API da OpenAI (`OPENAI_API_KEY`)

### 1.3 Dependências Python

As dependências estão em `requirements.txt` e incluem:

- `streamlit` (UI)
- `psycopg2-binary` + `pgvector` (acesso ao Postgres e tipo vetor)
- `python-dotenv` (carregar `.env`)
- `openai` + `langchain-openai` (LLM e embeddings)
- `langchain-*` (chunking e loaders)
- `pypdf`, `python-docx`, `markdown` (suporte a formatos)

---

## 2) Configuração (.env)

Crie um arquivo `.env` na raiz do projeto contendo:

```env
# OpenAI
OPENAI_API_KEY=coloque_sua_chave_aqui

# Postgres/pgvector
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag
DB_USER=rag
DB_PASSWORD=rag
```

Notas importantes:

- O projeto usa `python-dotenv` para carregar o `.env` automaticamente.
- Se você rodar o banco via Docker Compose (seção 4), ajuste os valores para bater com o `docker-compose.yml`.
- Quando o Postgres estiver em container e o Streamlit rodar localmente, normalmente `DB_HOST=localhost` e `DB_PORT` é a porta exposta no host.

---

## 3) Estrutura do projeto (passo a passo)

Visão geral das pastas:

```
.
├── main.py
├── docker-compose.yml
├── requirements.txt
└── app/
    ├── application/
    ├── domain/
    ├── infrastructure/
    └── presentation/
```

A organização segue um estilo “camadas”:

### 3.1 `main.py` (ponto de entrada)

Arquivo que inicializa a UI Streamlit:

- Renderiza título
- Monta sidebar com configurações
- Exibe chat
- Executa vetorização quando o usuário clica em **Iniciar Vetorização**
- Recebe pergunta, faz busca, monta contexto e chama o modelo LLM em streaming

Função relevante:

- `_build_context(search_results, max_chars=6000)`: limita o contexto enviado ao LLM por caracteres.

### 3.2 `app/presentation/` (UI)

Componentes de interface (Streamlit):

- `sidebar.py`: upload de arquivos, seleção de modelos e parâmetros (chunk, overlap, top_k, pesos da busca híbrida)
- `chat.py`: renderização do histórico de mensagens e “métricas” de tokens (atualmente fixas/dummy)
- `progress.py`: barra de progresso durante vetorização

### 3.3 `app/application/` (casos de uso)

Orquestração do fluxo de negócio:

- `vectorization.py`
  - Lê arquivos (PDF/DOCX/MD/TXT)
  - Se for `.zip`, extrai e processa arquivos internos
  - Faz chunking (`domain/chunking.py`)
  - Gera embeddings (`infrastructure/embeddings.py`)
  - Persiste no banco (`infrastructure/database.py`)

- `search.py`
  - Seleciona a estratégia de busca (vetorial/semântica/híbrida)
  - Executa a busca via `domain/search.py`

### 3.4 `app/domain/` (regras e modelos)

Regras “puras” e estratégias:

- `chunking.py`: define como o texto é dividido (usa `RecursiveCharacterTextSplitter`)
- `search.py`: define o contrato `SearchStrategy`, `SearchResult` e implementações:
  - `VectorSearch`: embedding da query + busca por similaridade de cosseno no pgvector
  - `SemanticSearch`: busca full-text no Postgres (`tsvector`/`ts_rank`)
  - `HybridSearch`: combina resultados normalizados (vetorial + semântica) com `vector_weight` e filtra por `minimum_score`

### 3.5 `app/infrastructure/` (integrações)

Integrações externas:

- `database.py`
  - Conecta no Postgres usando variáveis de ambiente
  - Cria extensão `vector` e tabela `documents`
  - Insere documentos com embeddings
  - Implementa buscas:
    - `search_cosine_similarity` (pgvector)
    - `search_full_text` (full-text)

- `embeddings.py`
  - `get_embedding_model(model_name)` retorna `OpenAIEmbeddings`

- `llm.py`
  - `get_llm_client()` cria cliente OpenAI
  - `get_available_gpt_models()` lista modelos que começam com `gpt` (fallback se falhar)

---

## 4) Como executar (recomendado: banco no Docker + app local)

### 4.1 Subir Postgres com pgvector

1) Garanta que o `.env` exista (seção 2).

2) Suba os serviços:

```bash
docker compose up -d
```

Isso sobe:

- `db`: Postgres com pgvector
- `pgadmin`: interface web em `http://localhost:5050`

Credenciais do pgAdmin (fixas no compose):

- Email: `admin@admin.com`
- Senha: `admin`

### 4.2 Rodar a aplicação Streamlit (local)

1) Crie e ative um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate
```

2) Instale dependências:

```bash
pip install -r requirements.txt
```

3) Execute:

```bash
streamlit run main.py
```

Abra o navegador no endereço que o Streamlit imprimir (normalmente `http://localhost:8501`).

---

## 5) Como usar a aplicação

1) Na sidebar, selecione:

- Modelo LLM (lista dinamicamente via API; pode cair em fallback)
- Modelo de embedding
- Tipo de busca: **Vetorial**, **Semântica**, **Híbrida**
- Parâmetros: `chunk_size`, `overlap`, `top_k`

2) Faça upload de arquivos ou um `.zip`.

3) Clique em **Iniciar Vetorização**.

- A aplicação cria a tabela e insere os chunks com embeddings

4) No chat, digite uma pergunta.

- A aplicação busca os trechos mais relevantes
- Monta um “CONTEXTO” (até ~6000 caracteres)
- Envia ao LLM com `stream=True` e exibe a resposta incrementalmente

---

## 6) Requisitos funcionais (RF)

- RF01: Permitir upload de arquivos `.pdf`, `.docx`, `.txt`, `.md` e `.zip`.
- RF02: Extrair texto e dividir em chunks configuráveis.
- RF03: Gerar embeddings para chunks via OpenAI.
- RF04: Persistir chunks e embeddings no Postgres/pgvector.
- RF05: Permitir selecionar estratégia de recuperação (vetorial/semântica/híbrida).
- RF06: Receber pergunta do usuário e recuperar top-k trechos relevantes.
- RF07: Gerar resposta via LLM usando o contexto recuperado.

---

## 7) Estrutura do banco de dados

A tabela é criada automaticamente ao iniciar a vetorização (`create_table()`):

- Extensão: `CREATE EXTENSION IF NOT EXISTS vector;`
- Tabela: `documents`

Campos:

- `id SERIAL PRIMARY KEY`
- `content TEXT`
- `embedding VECTOR(1536)`

Observação: a dimensão do vetor está fixada em `1536`, compatível com os modelos de embedding oferecidos na UI.

---

## 8) Solução de problemas

### 8.1 Erro de conexão com banco

- Verifique se o Docker está rodando e se `docker compose ps` mostra o serviço `db` como “Up”.
- Confirme `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` no `.env`.

### 8.2 Erros com OpenAI

- Confirme `OPENAI_API_KEY` no `.env`.
- Verifique conectividade de rede.

### 8.3 Vetorização parece “travada”

- PDFs grandes e muitos chunks podem demorar.
- Ajuste `chunk_size`/`overlap` para reduzir quantidade de chunks.

---

## 9) Referência rápida de execução

```bash
# 1) Banco (docker)
docker compose up -d

# 2) App (local)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```
