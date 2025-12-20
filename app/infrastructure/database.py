import os
import psycopg2
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg2 import register_vector

load_dotenv()

def get_db_connection(register: bool = True):
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

    if register:
        register_vector(conn)
    return conn

def create_table():
    # Important: the 'vector' type only exists after the extension is created.
    # register_vector() will fail if called before that.
    conn = get_db_connection(register=False)
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    register_vector(conn)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding VECTOR(1536)
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

def insert_document(content, embedding):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
        (content, Vector(embedding)),
    )
    conn.commit()
    cursor.close()
    conn.close()


def truncate_documents_table() -> None:
    conn = get_db_connection(register=False)
    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
    conn.commit()
    cursor.close()
    conn.close()

def search_l2(query_embedding, limit=5):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, content, (embedding <-> %s) AS distance FROM documents ORDER BY distance ASC LIMIT %s",
        (Vector(query_embedding), limit),
    )
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

def search_inner_product(query_embedding, limit=5):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, content, ((embedding <#> %s) * -1) AS similarity FROM documents ORDER BY similarity DESC LIMIT %s",
        (Vector(query_embedding), limit),
    )
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

def search_cosine_similarity(query_embedding, limit=5):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, content, (1 - (embedding <=> %s)) AS similarity FROM documents ORDER BY similarity DESC LIMIT %s",
        (Vector(query_embedding), limit),
    )
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results


def search_full_text(query: str, limit: int = 5):
    """Simple semantic search using Postgres full-text search.

    Returns (id, content, score) where score is ts_rank.
    """
    conn = get_db_connection(register=False)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            id,
            content,
            ts_rank(to_tsvector('simple', content), plainto_tsquery('simple', %s)) AS rank
        FROM documents
        WHERE to_tsvector('simple', content) @@ plainto_tsquery('simple', %s)
        ORDER BY rank DESC
        LIMIT %s
        """,
        (query, query, limit),
    )
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results
