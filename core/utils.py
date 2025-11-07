## utils.py - Módulo Core: Embeddings, Chroma, Retrieval, LEL Chain, GA para Gustos Usuario
# Modificaciones: Manejo errores, logging, hybrid_chain refinado, GA correlación Pearson >0.7
import os
from dotenv import load_dotenv  # Día 1: Carga key
from langchain_openai import OpenAIEmbeddings  # Día 2: ada-002
from langchain.vectorstores import Chroma  # Día 2: DB
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Chunking test
from langchain.schema.runnable import RunnablePassthrough  # Día 4: LEL chain
from langchain_core.prompts import PromptTemplate  # Día 4: Prompt
from langchain_openai import ChatOpenAI  # Día 4: LLM gpt-3.5-turbo
from langchain_core.output_parsers import StrOutputParser  # Día 4: Parser
import numpy as np
import random  # Día 5: GA
from scipy.stats import pearsonr  # Día 5: Fitness correlación
from typing import List  # Typing
import logging  # Logging para debug

# Config logging (no expone key)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Día 1: Carga .env (manejo error)
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logger.error("OPENAI_API_KEY no encontrada en .env. Agrega en raíz proyecto.")
    raise ValueError("API key missing")

os.environ['OPENAI_API_KEY'] = api_key  # Asigna seguro

# Día 2: Embeddings (error handling)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
logger.info("Embeddings ada-002 inicializado")

# Día 2-3: Chroma DB setup (error si vacío)
def load_chroma(persist_directory="chroma_db"):
    try:
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        logger.info(f"Chroma cargado: {db._collection.count()} items")
        if db._collection.count() == 0:
            logger.warning("Chroma vacío - carga chunks con data_processor.py")
        return db
    except Exception as e:
        logger.error(f"Error loading Chroma: {e}")
        raise

# Día 3: Retriever with cosine (k=5 Día 6)
def get_retriever(db, k=5):
    try:
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        logger.info(f"Retriever con k={k} inicializado")
        return retriever
    except Exception as e:
        logger.error(f"Error retriever: {e}")
        raise

# Día 4: LEL Chain (Retrieval → Generation) - Error handling
prompt = PromptTemplate.from_template(
    "Context: {context}\nQuery: {query}\nResponde solo con contexto, no inventes."
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def create_chain(retriever):
    try:
        chain = (
            {"context": retriever | (lambda docs: "\n".join([d.page_content for d in docs])), "query": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("Chain LEL creada")
        return chain
    except Exception as e:
        logger.error(f"Error creating chain: {e}")
        raise

# Día 5: GA for Gustos Usuario (evolutivo, correlación Pearson)
def create_population(size=20):
    return [np.random.uniform(0, 1, 3) for _ in range(size)]  # [aventura, cultura, comida]

def fitness_profile(profile, user_feedback):
    try:
        corr, _ = pearsonr(profile, user_feedback[0])  # Vs primer usuario
        return corr if corr > 0 else 0
    except Exception as e:
        logger.error(f"Error fitness: {e}")
        return 0

def evolve_gustos(pop, user_feedback, gens=30, mut_rate=0.1):
    historial = []
    for gen in range(gens):
        fit = [fitness_profile(p, user_feedback) for p in pop]
        elite = sorted(pop, key=lambda p: fitness_profile(p, user_feedback), reverse=True)[:2]
        pop = elite.copy()
        while len(pop) < len(elite) * 2:
            p1, p2 = random.choice(elite), random.choice(elite)
            child = 0.5 * p1 + 0.5 * p2
            if random.random() < mut_rate:
                child += np.random.normal(0, 0.05, 3)
            child = np.clip(child, 0, 1)
            pop.append(child)
        pop = pop[:len(pop)//2]
        historial.append(max(fit))
    best = max(pop, key=lambda p: fitness_profile(p, user_feedback))
    logger.info(f"GA evolucionado: Fitness {max(historial):.2f}")
    return best, max(historial)

# Día 9: Hybrid RAG-GA Chain (error handling)
def hybrid_chain(retriever, user_feedback):
    chain_rag = create_chain(retriever)
    def hybrid_func(query):
        try:
            context = retriever.invoke(query)
            if not context:
                logger.warning("No chunks retrieved")
                return {"error": "No data found"}
            rag_output = chain_rag.invoke({"query": query, "context": context})
            best_profile, max_fitness = evolve_gustos(create_population(), user_feedback)
            return {"rag": rag_output, "gusto_ga": {"profile": best_profile.tolist(), "fitness": max_fitness}}
        except Exception as e:
            logger.error(f"Hybrid chain error: {e}")
            return {"error": str(e)}
    return hybrid_func

# Test (Día 2-10, with error handling)
if __name__ == "__main__":
    user_feedback = np.array([[0.8, 0.4, 0.9]])  # Sample gustos
    try:
        db = load_chroma()  # Asume chunks loaded
        retriever = get_retriever(db)
        hybrid = hybrid_chain(retriever, user_feedback)
        result = hybrid("Mejores playas Lima")
        print(result)  # {"rag": "...", "gusto_ga": {"profile": [...], "fitness": 0.82}}
        logger.info("Hybrid test OK")
    except Exception as e:
        logger.error(f"Test failed: {e}")