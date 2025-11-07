# data_processor.py - Módulo 1: Procesamiento CSV con Header Personalizado (REGIÓN, PROVINCIA, DISTRITO, NOMBRE DEL RECURSO, SUB TIPO CATEGORÍA, URL, LATITUD, LONGITUD)
# Alineado cronograma: Día 1 loader; Día 2 splitter; Día 3 full process + populate Chroma
from langchain.document_loaders import CSVLoader  # Loader base CSV
from langchain.text_splitter import RecursiveTextSplitter  # Chunking
from langchain.schema import Document  # Para docs
import pandas as pd  # Lectura header custom
import json  # Guardado chunks JSON
import os  # Carpetas
import logging  # Debug

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_chunk_csv(csv_path='lugares_lima.csv'):
    """
    Lee CSV con header específico, genera chunks de NOMBRE DEL RECURSO + SUB TIPO CATEGORÍA + REGIÓN/PROVINCIA/DISTRITO.
    Guarda chunks en 'chunks/' como JSON, y pobla Chroma DB.
    """
    try:
        # Paso 1: Lectura CSV con pandas (header custom)
        df = pd.read_csv(csv_path)
        logger.info(f"CSV cargado: {len(df)} filas, columnas: {df.columns.tolist()}")
        
        if len(df) == 0:
            logger.error("CSV vacío – agrega filas")
            return [], None
        
        # Paso 2: Crear documentos LangChain (contenido: Nombre + Sub Tipo + Región/Provincia/Distrito)
        docs = []
        for index, row in df.iterrows():
            content = (
                f"REGIÓN: {row['REGIÓN']}. PROVINCIA: {row['PROVINCIA']}. DISTRITO: {row['DISTRITO']}.\n"
                f"NOMBRE DEL RECURSO: {row['NOMBRE DEL RECURSO']}.\n"
                f"SUB TIPO CATEGORÍA: {row['SUB TIPO CATEGORÍA']}."
            )
            metadata = {
                "source": "CSV",
                "url": row['URL'],
                "latitud": row['LATITUD'],
                "longitud": row['LONGITUD'],
                "row": index
            }
            docs.append(Document(page_content=content, metadata=metadata))
        
        logger.info(f"Documentos creados: {len(docs)}")
        
        # Paso 3: Chunking (Día 2: 1500/200)
        splitter = RecursiveOverlapTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        logger.info(f"Chunks generados: {len(chunks)}")
        
        if len(chunks) == 0:
            logger.error("No chunks – revisa content length")
            return [], None
        
        # Paso 4: Guardar chunks en 'chunks/' como JSON (persistencia)
        os.makedirs('chunks', exist_ok=True)
        for i, chunk in enumerate(chunks):
            filename = f'chunks/chunk_{i+1}.json'
            data = {
                'content': chunk.page_content,
                'metadata': chunk.metadata
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Chunks guardados en 'chunks/': {len(chunks)} archivos")
        
        # Paso 5: Poblar Chroma DB (Día 3: full process)
        from core.utils import embeddings, load_chroma  # Integra con utils.py
        db = load_chroma()
        db.add_documents(chunks)
        db.persist()
        logger.info(f"Chroma poblado: {db._collection.count()} items")
        
        return chunks, db  # Retorna para test/integración
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        return [], None

# Test (Día 1-3: Run para poblar)
if __name__ == "__main__":
    chunks, db = load_and_chunk_csv()
    if chunks:
        print(f"Test OK: {len(chunks)} chunks, DB {db._collection.count() if db else 0}")
    else:
        print("Test FAILED: No chunks generated – revisa CSV")