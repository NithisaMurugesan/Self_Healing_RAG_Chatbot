from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate # allows dynamic inputs of {contexts} and {questions}
from langchain_core.output_parsers import StrOutputParser # gives the output as plain string and not as a format
from langchain_groq import ChatGroq # Connects LangChain to Groq's LLM API.
import time
import json
from dotenv import load_dotenv

load_dotenv()

# joins retrieved chunks into a single string 
def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        formatted.append(f"[Source:{source}, Page:{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)
    
def build_rag(pdf_path):
    # LOAD PDF
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    print(f"Loaded {len(pages)} pages from the pdf")
    print("\n ---- First pafe preview(first 500 characters) -----")
    print(pages[0].page_content[:500])
    
    # CHUNKS CREATION
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 600,
        chunk_overlap = 100,
        separators= ["\n\n", "\n", ",", " "],  # separates usign parahraphs or lines or sentences or words
    )
    
    chunks = splitter.split_documents(pages)
    print(f"Number of Chunks created: {len(chunks)}")
    
    # EMBEDDINGS CREATION
    
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2") # creating the embeddings
    vector_store = Chroma.from_documents(chunks,embeddings)  # store the chunks and embeddings in a vector store using chroma
    
    print(f"vector store ready. {vector_store._collection.count()} vectors stored.")
    
    # RETRIEVAL CHAIN
    
    # vector retriever (retriever 1)
    vector_retriever = vector_store.as_retriever(search_kwargs={
        "k": 3, # k=3 means return top 3 most relevant chunks for the query 
    })   
    
    # bm25 retriever (retriever 2)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3  # Top 3 matching chunks
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_retriever
        ],
        weights=[0.5,0.5] # ensemble_retriever merges both bm25 and vector retrievers retrieved chunks with 50% importance each therefore giving us combined ranking
    )
    
    model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base" # this model scores the chunks retrieved according to relevance with the question
    )
    
    compressor = CrossEncoderReranker(
        top_n=3, # keeps only the top 3 reranked most relevant chunks - this is called compression (compressing total retrieved chunks to just 3 chunks)
        model=model 
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=ensemble_retriever,
        base_compressor=compressor
    )
    
    return vector_store, compression_retriever

# RAG Pipeline and Monitoring

def rag_pipeline(question, compression_retriever):
    
    # query rewrite and monitoring
    rewrite_start = time.time()
    rewritten_question = query_rewriter.invoke(
        {"question": question}
    )
    rewrite_end = time.time()
    
    # retrieval and retrieval monitoring
    retrieval_start = time.time()
    docs = compression_retriever.invoke(
        rewritten_question
    )
    context = format_docs(docs)
    retrieval_end = time.time()
    
    # llm answer generation and monitoring 
    generation_start = time.time()
    answer = llm.invoke(
        prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )
    )
    answer_text = answer.content
    generation_end = time.time()
    return (answer_text, context, rewritten_question, docs,rewrite_end - rewrite_start, retrieval_end - retrieval_start, generation_end - generation_start)

# Relevancy evaluation
def evaluate_answer(context, answer):
    result = hallucination_checker.invoke(
        {
            "context": context,
            "answer": answer
        }
    )
    evaluation = json.loads(result)
    score = evaluation["score"]
    reason = evaluation["reason"]
    return score, reason

SYSTEM_PROMPT = """
You are a helpful telecom assistant.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.

When using information from the context,
include the corresponding source and page number
at the end of the relevant statement.

Example:
International roaming allows users to connect to partner networks abroad.
[Source: telecom_guide.pdf, Page: 7]

Context:
{context}

"""

HALLUCINATION_PROMPT = """
You are evaluating whether an answer is grounded
in the provided context.

Context:
{context}

Answer:
{answer}

Give a faithfulness score from 0 to 100.

100 = Every claim is supported.
80 = Minor unsupported details.
60 = Some unsupported claims.
40 = Significant hallucinations.
20 = Mostly hallucinated.
0 = Completely unsupported.

Return JSON only:

{{
    "score": number,
    "reason": "short explanation"
}}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

hallucination_prompt = ChatPromptTemplate.from_template(
    HALLUCINATION_PROMPT
)

# llm called qwen from chatgroq

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    reasoning_format="parsed", # gives only the final answer excluding the thinking process
)

rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the user's question to improve document retrieval.

Rules:
- Return ONLY the rewritten question.
- Do not explain.
- Do not provide bullet points.
- Do not provide analysis.
- Return one sentence.

Question:
{question}
""")

hallucination_checker = (
    hallucination_prompt
    | llm
    | StrOutputParser()
)

query_rewriter = (
    rewrite_prompt
    | llm
    | StrOutputParser()
)


def ask_question(question, vector_store, compression_retriever):
    total_start = time.time()

    # ACCESSING NECESSARY PARAMETERS
    (answer_text, context, rewritten_question, docs, rewrite_time, retrieval_time, generation_time) = rag_pipeline(
        question,
        compression_retriever
    )

    # EVALUATION
    score, reason = evaluate_answer(
        context,
        answer_text
    )

    print(f"\nFaithfulness Score: {score}")
    print(f"Reason: {reason}")

    # SELF HEALING
    retry_count = 0
    while score < 80 and retry_count < 2:
        print("\nHallucination Detected")
        print(f"Retry #{retry_count + 1}")
        larger_retriever = vector_store.as_retriever(
            search_kwargs={"k": 6}
        )
        (answer_text, context, rewritten_question, docs, rewrite_time, retrieval_time, generation_time) = rag_pipeline(
            question,
            larger_retriever
        )
        score, reason = evaluate_answer(context,answer_text)
        print(f"\nFaithfulness Score: {score}")
        print(f"Reason: {reason}")
        retry_count += 1

    # MONITORING
    total_end = time.time()
    print("-" * 50)

    print("Original Question:")
    print(question)

    print("\nRewritten Question:")
    print(rewritten_question)

    print(f"\nChunks Retrieved: {len(docs)}")

    print(f"\nFaithfulness Score: {score}")
    print(f"Reason: {reason}")

    print( f"\nTotal Time:{total_end - total_start:.3f}s")
    print(f"Rewrite Time: {rewrite_time:.3f}s")
    print(f"Retrieval Time: {retrieval_time:.3f}s")
    print(f"Generation Time: {generation_time:.3f}s"
         )
    print("-" * 50)
    return {
    "answer": answer_text,
    "faithfulness_score": score,
    "faithfulness_reason": reason,
    "retry_count": retry_count,
    "rewrite_time": rewrite_time,
    "retrieval_time": retrieval_time,
    "generation_time": generation_time,
    "rewritten_question": rewritten_question
}
