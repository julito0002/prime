# data_processor.py - Módulo 1: Procesamiento CSV con Header Específico (REGIÓN, PROVINCIA, DISTRITO, NOMBRE DEL RECURSO, SUB TIPO CATEGORÍA, URL, LATITUD, LONGITUD)
# Alineado cronograma: Día 1 loader; Día 2 splitter; Día 3 full process
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveTextSplitter
from langchain.schema import Document
import pandas as pd  # Para lectura header
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_chunk_csv(csv_path='lugares_lima.csv'):
    """
    Lee CSV con header específico, genera chunks de 'NOMBRE DEL RECURSO' + 'SUB TIPO CATEGORÍA'.
    """
    try:
        # Lee CSV con pandas para header específico
        df = pd.read_csv(csv_path)
        logger.info(f"CSV cargado: {len(df)} filas, columnas: {df.columns.tolist()}")
        
        # Crea documentos: Nombre + Sub Tipo como contenido (ignora Lat/Long para chunks, usa en GA)
        docs = []
        for index, row in df.iterrows():
            content = f"NOMBRE DEL RECURSO: {row['NOMBRE DEL RECURSO']}. SUB TIPO CATEGORÍA: {row['SUB TIPO CATEGORÍA']}. REGIÓN: {row['REGIÓN']}, PROVINCIA: {row['PROVINCIA']}, DISTRITO: {row['DISTRITO']}."
            metadata = {"url": row['URL'], "lat": row['LATITUD'], "long": row['LONGITUD']}
            docs.append(Document(page_content=content, metadata=metadata))
        
        # Chunking (Día 2: 1500/200)
        splitter = RecursiveTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        logger.info(f"Chunks generados: {len(chunks)}")
        
        # Pobla Chroma (Día 3: full process)
        from core.utils import embeddings, load_chroma
        db = load_chroma()
        db.add_documents(chunks)
        db.persist()
        logger.info(f"Chroma poblado: {db._collection.count()} items")
        
        return chunks, db  # Retorna para test
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        return [], None

# Test (Día 1-3)
if __name__ == "__main__":
    chunks, db = load_and_chunk_csv()
    print(f"Test OK: {len(chunks)} chunks, DB {db._collection.count() if db else 0}")