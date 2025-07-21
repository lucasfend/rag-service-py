import os
import logging
from typing import Dict, List, Any
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from text_processor import TextProcessor
import re

load_dotenv()
logger = logging.getLogger(__name__)

class DatabaseService:
    """
    Serviço para interagir com o MongoDB e realizar buscas semânticas
    """ 
    model = SentenceTransformer('all-MiniLM-L6-v2')
    kw_model = KeyBERT(model)

    @staticmethod
    def extract_keywords(text: str, top_n: int = 3) -> list:
        keywords = DatabaseService.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
        return [kw[0] for kw in keywords]

    def __init__(self):
        # Configurações do MongoDB
        self.connection_string = os.getenv('MONGODB_URI')
        self.database_name = os.getenv('MONGODB_DATABASE', 'academic_db')
        self.collection_name = os.getenv('MONGODB_COLLECTION', 'documents')
        
        if not self.connection_string:
            raise ValueError("MONGODB_URI não encontrada no .env")
        
        # Conectar ao MongoDB
        self.client = MongoClient(self.connection_string)
        self.db = self.client[self.database_name]
        self.collection = self.db[self.collection_name]
        
        # Inicializar TF-IDF para busca semântica
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # Cache para evitar reprocessamento
        self._documents_cache = None
        self._tfidf_matrix = None
        
        logger.info("DatabaseService inicializado com sucesso")
    
    def search_documents(self, query: str, filters: dict = None, limit: int = 5) -> list:
        try:
            mongo_filters = {}

            # Usar busca $text para aproveitamento do índice textual
            if query and query.strip():
                mongo_filters = {"$text": {"$search": query}}

            print(f"[DEBUG] Filtros utilizados: {mongo_filters}")

            cursor = self.collection.find(
                mongo_filters,
                {
                    'subject': 1,
                    'className': 1,
                    'tutor': 1,
                    'document': 1
#                    'score': {'$meta': "textScore"}  # opcional para ordenar por relevância
                }
            ).limit(limit * 3)

            documents = list(cursor)
            print(f"[DEBUG] Documentos retornados da base: {len(documents)}")

            # Ordenar por score do texto se disponível
            if documents and 'score' in documents[0]:
                documents.sort(key=lambda d: d.get('score', 0), reverse=True)

            # Se nenhum documento foi encontrado, usar fallback simples por substring
            if not documents and query:
                print("[DEBUG] Fallback: busca por substring simples")
                query_lower = query.lower()
                substring_conditions = [
                    {"subject": {"$regex": query_lower, "$options": "i"}},
                    {"tutor": {"$regex": query_lower, "$options": "i"}},
                    {"className": {"$regex": query_lower, "$options": "i"}},
                    {"document": {"$regex": query_lower, "$options": "i"}}
                ]
                mongo_filters = {"$or": substring_conditions}

                cursor = self.collection.find(
                    mongo_filters,
                    {
                        'subject': 1,
                        'className': 1,
                        'tutor': 1,
                        'document': 1
                    }
                ).limit(limit * 3)

                documents = list(cursor)
                print(f"[DEBUG] Documentos encontrados com fallback: {len(documents)}")

            if not documents:
                return []

            if len(documents) > 1:
                ranked_docs = self._rank_documents_by_similarity(query, documents)
                return ranked_docs[:limit]
            else:
                documents[0]['relevance_score'] = 1.0
                return documents

        except Exception as e:
            logger.error(f"Erro na busca de documentos: {str(e)}")
            raise


    def _rank_documents_by_similarity(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Ranqueia documentos por similaridade semântica usando TF-IDF
        """
        try:
            # Preparar textos para análise
            doc_texts = []
            for doc in documents:
                # Combinar todos os campos textuais relevantes
                combined_text = f"{doc.get('subject', '')} {doc.get('tutor', '')} {doc.get('className', '')} {doc.get('document', '')}"
                doc_texts.append(combined_text)
            
            # Adicionar a query aos textos
            all_texts = doc_texts + [query]
            
            # Calcular TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calcular similaridade
            query_vector = tfidf_matrix[-1]  # Último elemento é a query
            doc_vectors = tfidf_matrix[:-1]  # Todos exceto o último
            
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            # Adicionar scores aos documentos
            for i, doc in enumerate(documents):
                doc['relevance_score'] = float(similarities[i])
           
            print("Documentos encontrados:", len(documents))
            for doc in documents:
                print("DOC:", doc)

            # Ordenar por relevância
            ranked_docs = sorted(documents, key=lambda x: x['relevance_score'], reverse=True)
            
            # Filtrar documentos com score muito baixo (mais permissivo)
            min_score = 0.0
            filtered_docs = [doc for doc in ranked_docs if doc['relevance_score'] >= min_score]
            
            logger.info(f"Documentos ranqueados: {len(filtered_docs)} de {len(documents)}")
            
            return filtered_docs if filtered_docs else ranked_docs[:1]  # Retornar pelo menos um
            
        except Exception as e:
            logger.error(f"Erro no ranking de documentos: {str(e)}")
            # Em caso de erro, retornar documentos sem ranking
            for doc in documents:
                doc['relevance_score'] = 0.5
            return documents
    
    def get_document_count(self, filters: Dict[str, Any] = None) -> int:
        """
        Conta o número de documentos que atendem aos filtros
        """
        try:
            mongo_filters = {}
            
            if filters:
                for key, value in filters.items():
                    if value and value.strip():
                        mongo_filters[key] = {"$regex": value.strip(), "$options": "i"}
            
            count = self.collection.count_documents(mongo_filters)
            return count
            
        except Exception as e:
            logger.error(f"Erro ao contar documentos: {str(e)}")
            return 0
    
    def test_connection(self) -> bool:
        """
        Testa a conexão com o MongoDB
        """
        try:
            # Tentar fazer uma operação simples
            self.client.admin.command('ping')
            logger.info("Conexão com MongoDB testada com sucesso")
            return True
        except Exception as e:
            logger.error(f"Erro na conexão com MongoDB: {str(e)}")
            return False
    
    def __del__(self):
        """
        Fechar conexão ao destruir o objeto
        """
        if hasattr(self, 'client'):
            self.client.close()
