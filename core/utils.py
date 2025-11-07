# utils.py - Módulo Core: Embeddings, Chroma, Retrieval, LEL Chain, GA para Gustos Usuario
# Alineado con cronograma: Día 1 .env; Día 2 embeddings; Día 3 retriever; Día 4 chain; Día 5 GA retrieval; Día 6 k=5; Día 7 elitismo; Día 9 GA en LEL
import os
from dotenv import load_dotenv  # Día 1: Carga key
from langchain_openai import OpenAIEmbeddings  # Día 2: ada-002
from langchain.vectorstores import Chroma  # Día 2: DB
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Chunking from data_processor, but used here for test
from langchain.schema.runnable import RunnablePassthrough  # Día 4: LEL chain
from langchain_core.prompts import PromptTemplate  # Día 4: Prompt
from langchain_openai import ChatOpenAI  # Día 4: LLM gpt-3.5-turbo
from langchain_core.output_parsers import StrOutputParser  # Día 4: Parser
import numpy as np
import random  # Día 5: GA
from scipy.stats import pearsonr  # Día 5: Fitness correlación
from typing import List  # Typing for GA

# Día 1: Carga .env
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')  # Seguridad key

# Día 2: Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # 1536 dims

# Día 2-3: Chroma DB setup
def load_chroma(persist_directory="chroma_db"):  # Día 2: Init vacío
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Día 3: Retriever with cosine
def get_retriever(db, k=5):  # Día 6: k=5
    return db.as_retriever(search_type="similarity", search_kwargs={"k": k})  # Cosine default

# Día 4: LEL Chain (Retrieval → Generation)
prompt = PromptTemplate.from_template(
    "Context: {context}\nQuery: {query}\nResponde solo con contexto, no inventes."
)  # Día 4: Prompt template

llm = ChatOpenChain(model="gpt-3.5-turbo", temperature=0)  # Día 4: LLM

def create_chain(retriever):
    chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )  # Día 4: LEL encadenado
    return chain

# Día 5: GA for Gustos Usuario (evolutivo, no rutas)
def create_population(size=20):  # Día 5: Población perfiles gustos
    return [np.random.uniform(0, 1, 3) for _ in range(size)]  # [aventura, cultura, comida]

def fitness_profile(profile, user_feedback):  # Día 5: Fitness correlación
    corr, _ = pearsonr(profile, user_feedback[0])  # Vs primer usuario
    return corr if corr > 0 else 0  # Maximiza

def evolve_gustos(pop, user_feedback, gens=30, mut_rate=0.1):  # Día 7: Elitismo
    historial = []
    for gen in range(gens):
        fit = [fitness_profile(p, user_feedback) for p in pop]
        # Elitismo top 2
        elite = sorted(pop, key=lambda p: fitness_profile(p, user_feedback), reverse=True)[:2]
        pop = elite
        while len(pop) < len(elite) * 2:  # Cruce
            p1, p2 = random.choice(elite), random.choice(elite)
            child = 0.5 * p1 + 0.5 * p2
            if random.random() < mut_rate:  # Mutación
                child += np.random.normal(0, 0.05, 3)
            child = np.clip(child, 0, 1)
            pop.append(child)
        pop = pop[:len(pop)//2]  # Reduce población
        historial.append(max(fit))
    best = max(pop, key=lambda p: fitness_profile(p, user_feedback))
    return best, max(historial)  # Día 9: En chain LEL

# Día 9: Hybrid RAG-GA Chain
def hybrid_chain(retriever, user_feedback):
    chain_rag = create_chain(retriever)  # RAG part
    def hybrid_func(query):
        context = retriever.invoke(query)  # Retrieval
        rag_output = chain_rag.invoke({"query": query, "context": context})  # Generation
        best_profile, max_fitness = evolve_gustos(create_population(), user_feedback)  # GA evolution
        return {"rag": rag_output, "gusto_ga": {"profile": best_profile.tolist(), "fitness": max_fitness}}  # Pydantic ready
    return hybrid_func  # LEL-like function

# Test (Día 2-9)
if __name__ == "__main__":
    user_feedback = np.array([[0.8, 0.4, 0.9]])  # Sample gustos usuario
    db = load_chroma()  # Asume chunks loaded
    retriever = get_retriever(db)
    hybrid = hybrid_chain(retriever, user_feedback)
    result = hybrid("Mejores playas Lima")
    print(result)  # {"rag": "Recomendación...", "gusto_ga": {"profile": [0.75, 0.45, 0.85], "fitness": 0.82}}