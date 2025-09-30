

# rag_app_updated.py — Fast Multi-Activity Search
# - Fast mode (fuzzy + vector) for 1–2 sec responses
# - Optional LLM refinement (slower but more accurate)
# - Handles multiple activities at once

import os
import re
import json
import ast
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from rapidfuzz import fuzz


# ---------------- HELPERS ----------------

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"&", "and", text)       # replace & with "and"
    text = re.sub(r"\s+", " ", text)       # collapse multiple spaces
    return text


def fuzzy_exact_match(query: str, df: pd.DataFrame, threshold: int = 95):
    """Return the best-matching row (Series) and score if score >= threshold."""
    best_row = None
    best_score = 0
    qn = normalize_text(query)
    for _, row in df.iterrows():
        candidate = normalize_text(row.get("activity name", ""))
        score = fuzz.ratio(qn, candidate)
        if score > best_score:
            best_score = score
            best_row = row
    if best_score >= threshold:
        return best_row, int(best_score)
    return None, None


def parse_json_like(text: str):
    """Try to parse a JSON object from arbitrary LLM text with multiple fallbacks."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return ast.literal_eval(candidate)
            except Exception:
                return {"llm_raw": text}
    return {"llm_raw": text}


# ---------------- CONFIG ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

VECTOR_STORE_DIR = "vectorstores"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# SHEETS_ORDER = ["dafza.xlsx", "meydan.xlsx", "spc.xlsx", "isic.xlsx"]
SHEETS_ORDER = ["all_activities_sheet.xlsx"]  # Single combined sheet for simplicity


# ---------------- LOAD SHEETS ----------------
@st.cache_data
def load_sheets():
    """Load all Excel sheets into DataFrames with normalized column names."""
    dataframes = {}
    for sheet in SHEETS_ORDER:
        df = pd.read_excel(sheet)
        df.columns = [c.strip().lower() for c in df.columns]
        if "activity name" not in df.columns or "class" not in df.columns:
            raise ValueError(f"Sheet {sheet} must contain 'Activity Name' and 'Class' columns.")
        dataframes[sheet] = df
    return dataframes


# ---------------- VECTOR STORE ----------------
@st.cache_resource
def load_or_create_vectorstore(sheet_name: str, df: pd.DataFrame):
    """Create or load FAISS vector store for a given sheet."""
    store_path = os.path.join(VECTOR_STORE_DIR, f"{sheet_name}_faiss")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if os.path.exists(store_path):
        vectorstore = FAISS.load_local(
            store_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        docs = [
            Document(page_content=str(row["activity name"]), metadata={"class": str(row["class"])} )
            for _, row in df.iterrows()
        ]
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(store_path)
    return vectorstore


# ---------------- SEARCH LOGIC ----------------

def search_activity(query: str, dataframes: dict, vectorstores: dict, use_llm: bool = False):
    """
    Search for a single activity.
    If use_llm=False -> fast (1–2 sec).
    If use_llm=True  -> slower, with LLM refinement.
    """

    # 1) Fuzzy exact match (fastest)
    for sheet in SHEETS_ORDER:
        df = dataframes[sheet]
        row, score = fuzzy_exact_match(query, df, threshold=95)
        if row is not None:
            return {
                "source": sheet,
                "query" : query,
                "activity": row["activity name"],
                "code": row["class"],
                "method": f"Fuzzy Exact Match ({score}%)",
                "score": score,
            }

    # 2) Vector similarity search
    all_candidates = []
    for sheet in SHEETS_ORDER:
        docs = vectorstores[sheet].similarity_search(query, k=5)
        for d in docs:
            all_candidates.append({
                "query" : query,
                "sheet": sheet,
                "activity": d.page_content,
                "code": d.metadata.get("class")
            })

    # Fast mode: return first/best candidate directly
    if not use_llm:
        if all_candidates:
            best = all_candidates[0]
            return {
                "query" : query,
                "source": best["sheet"],
                "activity": best["activity"],
                "code": best["code"],
                "method": "Vector (fast mode)",
            }
        return {"source": None, "activity": None, "code": None, "method": "No match"}

    # 3) LLM refinement (slower)
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0)
    prompt = f"""
    You are given:
    - A user query (business activity name).
    - A list of candidate activities retrieved from a database (with sheet, activity, and code).

    Your task:
    1. Pick the single best candidate ONLY from the list.
    2. Do not invent or create new activities. 
    3. Prefer semantic matches (e.g., "hydropanels" → "solar equipment trading", not "swimming pools").
    4. Act as a strict classifier who knows the domain well as ISIC codes.
    5. Return ONLY a JSON object with these keys:
    - query
    - sheet (string )
    - activity (string )
    - code (string)
    - reason (short explanation)

    User query: "{query}"

    Candidates:{all_candidates}
    """
    response = llm.invoke(prompt)
    parsed = parse_json_like(response.content)

    if isinstance(parsed, dict) and all(k in parsed for k in ("sheet", "activity", "code")):
        return {
            "query" : query,
            "source": parsed.get("sheet"),
            "activity": parsed.get("activity"),
            "code": parsed.get("code"),
            "method": "Vector + LLM",
            "reason": parsed.get("reason"),
            "llm_raw": response.content,
        }

    return {"source": None, "activity": None, "code": None, "method": "No match", "llm_raw": response.content}


def search_multiple_activities(queries, dataframes, vectorstores, use_llm=False):
    """Search codes for multiple activity queries."""
    results = []
    for q in queries:
        q = q.strip()
        if not q:
            continue
        result = search_activity(q, dataframes, vectorstores, use_llm=use_llm)
        results.append(result)
    return results


# ---------------- STREAMLIT APP ----------------

def main():
    st.title("Activity Code Finder (RAG)")
    st.write("Search activity codes from consolidated sheets. Supports multiple activities at once.")

    try:
        dataframes = load_sheets()
    except Exception as e:
        st.error(f"Failed to load sheets: {e}")
        return

    vectorstores = {sheet: load_or_create_vectorstore(sheet, df) for sheet, df in dataframes.items()}

    query = st.text_area("Enter one or multiple activities (each on a newline):")
    # use_llm = st.checkbox("Use LLM refinement (slower, more accurate)", value=False)

    if st.button("Search") and query:
        queries = [q.strip() for q in re.split(r"[\n]", query) if q.strip()]
        results = search_multiple_activities(queries, dataframes, vectorstores, use_llm=True)

        for res in results:
            st.subheader(res.get("activity") or "No match")
            st.write("**Query:**", res.get("query"))
            st.write("**Code:**", res.get("code"))
            # st.write("**Source sheet:**", res.get("source"))
            # st.write("**Method:**", res.get("method"))
            if res.get("score") is not None:
                st.write("**Score:**", res.get("score"))
            if res.get("reason"):
                st.write("**Reason:**", res.get("reason"))


if __name__ == "__main__":
    main()
