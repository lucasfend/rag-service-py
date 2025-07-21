import os
import logging
from typing import Dict, List, Any
from database_service import DatabaseService
from openai_service import OpenAIService
from text_processor import TextProcessor
import re

logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Sistema RAG (Retrieval-Augmented Generation) para buscar e gerar respostas
    """
    
    def __init__(self):
        self.db_service = DatabaseService()
        self.openai_service = OpenAIService()
        self.text_processor = TextProcessor()
        
        # Configurações do sistema
        self.max_context_length = int(os.getenv('MAX_CONTEXT_LENGTH', '3000'))
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.1'))  # Mais permissivo
        self.max_results = int(os.getenv('MAX_RESULTS', '5'))
        
    def process_question(self, question: str, filters: dict = None) -> dict:
        try:
            logger.info("Buscando documentos relevantes...")
            relevant_docs = self.db_service.search_documents(question, filters=filters, limit=self.max_results)

            if not relevant_docs:
                # Tentativa adicional com busca mais ampla
                logger.info("Primeira busca sem resultados, tentando busca mais ampla...")
                relevant_docs = self._fallback_search(question)

            if not relevant_docs:
                return {
                    'answer': "Desculpe, não encontrei informações relevantes para responder sua pergunta. Tente reformular a pergunta ou usar termos mais específicos.",
                    'context_used': "",
                    'sources': [],
                    'tokens_used': 0
                }

            context = self._prepare_context(relevant_docs)
            prompt = self._generate_prompt(question, context)
            logger.info("Enviando para OpenAI...")
            openai_response = self.openai_service.generate_response(prompt)

            sources = self._prepare_sources(relevant_docs)

            return {
                'answer': openai_response['response'],
                'context_used': context,
                'sources': sources,
                'tokens_used': openai_response['tokens_used']
            }

        except Exception as e:
            logger.error(f"Erro no processamento RAG: {str(e)}")
            raise

    def _fallback_search(self, question: str) -> List[Dict]:
        """
        Busca de fallback mais ampla quando a busca principal não retorna resultados
        """
        try:
            # Extrair apenas palavras significativas
            words = [word.lower() for word in question.split() if len(word) > 3]
            
            if not words:
                return []
            
            # Tentar busca com cada palavra individualmente
            all_docs = []
            
            for word in words:
                try:
                    cursor = self.db_service.collection.find(
                        {
                            "$or": [
                                {"subject": {"$regex": f".*{re.escape(word)}.*", "$options": "i"}},
                                {"document": {"$regex": f".*{re.escape(word)}.*", "$options": "i"}},
                                {"tutor": {"$regex": f".*{re.escape(word)}.*", "$options": "i"}},
                                {"className": {"$regex": f".*{re.escape(word)}.*", "$options": "i"}}
                            ]
                        },
                        {
                            'subject': 1,
                            'className': 1,
                            'tutor': 1,
                            'document': 1
                        }
                    ).limit(3)
                    
                    docs = list(cursor)
                    all_docs.extend(docs)
                    
                except Exception as e:
                    logger.warning(f"Erro na busca de fallback para palavra '{word}': {e}")
                    continue
            
            # Remover duplicatas
            unique_docs = {}
            for doc in all_docs:
                doc_id = str(doc.get('_id'))
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = doc
            
            result_docs = list(unique_docs.values())
            
            if result_docs:
                logger.info(f"Busca de fallback encontrou {len(result_docs)} documentos")
                # Adicionar score básico
                for doc in result_docs:
                    doc['relevance_score'] = 0.3
            
            return result_docs[:self.max_results]
            
        except Exception as e:
            logger.error(f"Erro na busca de fallback: {e}")
            return []

    def _prepare_context(self, documents: List[Dict]) -> str:
        """
        Prepara o contexto a partir dos documentos recuperados
        """
        context_parts = []
        current_length = 0
        
        for doc in documents:
            # Criar contexto estruturado com metadados
            doc_context = f"""
**Disciplina:** {doc.get('subject', 'N/A')}
**Professor:** {doc.get('tutor', 'N/A')}
**Turma:** {doc.get('className', 'N/A')}
**Conteúdo:**
{doc.get('document', '')}
"""
            
            # Verificar se não excede o limite
            if current_length + len(doc_context) > self.max_context_length:
                # Truncar o documento se necessário
                remaining_space = self.max_context_length - current_length
                if remaining_space > 200:  # Mínimo para ser útil
                    doc_context = doc_context[:remaining_space] + "..."
                    context_parts.append(doc_context)
                break
            
            context_parts.append(doc_context)
            current_length += len(doc_context)
        
        return "\n---\n".join(context_parts)
    
    def _generate_prompt(self, question: str, context: str) -> str:
        """
        Gera o prompt estruturado para a OpenAI
        """
        prompt = f"""Você é um assistente acadêmico especializado.
Responda à pergunta abaixo com base no contexto extraído da base de dados.

**Instruções importantes:**
- Formate a resposta em **tópicos** sempre que possível
- Use **negrito** nos títulos principais
- Use <br/> para quebrar linhas entre seções
- Seja preciso e objetivo
- Cite as disciplinas e professores quando relevante
- Se não houver informação suficiente, seja honesto sobre isso

**Contexto útil:**
{context}

**Pergunta:**
{question}

**Resposta:**"""
        
        return prompt
    
    def _prepare_sources(self, documents: List[Dict]) -> List[Dict]:
        """
        Prepara as fontes dos documentos para referência
        """
        sources = []
        
        for doc in documents:
            source = {
                'id': str(doc.get('_id', '')),
                'subject': doc.get('subject', 'N/A'),
                'tutor': doc.get('tutor', 'N/A'),
                'className': doc.get('className', 'N/A'),
                'uploadedBy': doc.get('uploadedBy', 'N/A'),
                'relevance_score': doc.get('relevance_score', 0.0)
            }
            sources.append(source)
        
        return sources
