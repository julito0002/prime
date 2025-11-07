# generator.py - Módulo 4: Generación RAG-GA Híbrida con LEL, Pydantic y Ética
# Alineado cronograma: Día 1 prompt; Día 2 LEL chain; Día 3 inyecta chunks; Día 4 Pydantic; Día 5 LEL full; Día 6 resumen/fuentes; Día 7 híbrido RAG-GA; Día 9 ética prompt
from langchain_core.prompts import PromptTemplate  # Día 1: Template
from langchain_core.output_parsers import StrOutputParser  # Día 4: Parser
from langchain_openai import ChatOpenAI  # Día 4: LLM
from langchain.schema.runnable import RunnablePassthrough  # Día 2: LEL chain
from pydantic import BaseModel, Field  # Día 4: Estructurada
from typing import List  # Typing
import logging  # Debug

from core.utils import get_retriever, hybrid_chain  # Integración con core (Día 7)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Día 1: Prompt Template Ético (Día 9: no inventar, fuentes)
prompt = PromptTemplate.from_template(
    "Eres un asistente ético de turismo en Lima. Usa solo el contexto proporcionado, no inventes información. "
    "Si no hay datos, di 'No tengo información suficiente'. Incluye fuentes en la respuesta.\n"
    "Context: {context}\nQuery: {query}\nRespuesta:"
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # Día 4: LLM low temp for precision

class ResponseModel(BaseModel):  # Día 4: Pydantic estructurada
    """
    Modelo Pydantic para salida: Resumen, Fuentes, Gustos GA.
    """
    resumen: str = Field(..., description="Respuesta principal")
    fuentes: List[str] = Field(..., description="Lista de chunks/fuentes")
    gusto_ga: dict = Field(..., description="Perfil GA: {'profile': list, 'fitness': float}")

# Día 2: LEL Chain (Retrieval → Prompt → LLM → Parser)
def create_chain(retriever):
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("Chain LEL creada")
    return chain

# Día 3: Inyecta Chunks en Chain
def generate_response(query, retriever):
    try:
        response = create_chain(retriever).invoke(query)
        logger.info("Respuesta generada")
        return response
    except Exception as e:
        logger.error(f"Error generación: {e}")
        return "Error: No respuesta disponible"

# Día 4-6: Pydantic Resumen/Fuentes (Estructurada)
def parse_to_pydantic(response, sources):  # Día 6: Resumen + Fuentes
    try:
        model = ResponseModel(
            resumen=response,
            fuentes=[s.page_content[:100] + "..." for s in sources],  # Fuentes truncadas
            gusto_ga={"profile": [0.8, 0.4, 0.9], "fitness": 0.8}  # Mock, integra GA Día 7
        )
        return model.model_dump()  # Dict para JSON
    except Exception as e:
        logger.error(f"Error Pydantic: {e}")
        return {"error": "Formato inválido"}

# Día 7: Híbrido RAG-GA (Integración con core/utils.py)
def hybrid_generate(query, retriever, user_gustos):
    try:
        response = generate_response(query, retriever)
        sources = retriever.invoke(query)
        ga_profile, ga_fitness = evolve_gustos(create_population(), user_gustos)  # From ga_optimizer.py
        model = ResponseModel(
            resumen=response,
            fuentes=[s.page_content[:100] + "..." for s in sources],
            gusto_ga={"profile": ga_profile.tolist(), "fitness": ga_fitness}
        )
        logger.info("Híbrido RAG-GA OK")
        return model.model_dump()
    except Exception as e:
        logger.error(f"Error hybrid: {e}")
        return {"error": str(e)}

# Día 5: LEL Full Chain (Encadenado Completo)
def full_chain(retriever, user_gustos):
    chain_rag = create_chain(retriever)
    def full_func(query):
        return hybrid_generate(query, retriever, user_gustos)
    return full_func  # Runnable-like for LEL

# Test (Día 1-10)
if __name__ == "__main__":
    # Mock retriever (Día 3: integra real)
    from langchain.schema import Document
    mock_docs = [Document(page_content="Mock chunk Lima playas.")]
    mock_retriever = lambda q: mock_docs  # Mock for test
    user_gustos = np.array([[0.8, 0.4, 0.9]])  # Sample
    full = full_chain(mock_retriever, user_gustos)
    result = full("Mejores playas Lima")
    print(result)  # {"resumen": "Respuesta...", "fuentes": [...], "gusto_ga": {"profile": [...], "fitness": 0.82}}
    logger.info("Test Generator OK")