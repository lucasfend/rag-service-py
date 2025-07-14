import os
import logging
from typing import Dict, List, Any
from database_service import DatabaseService
from openai_service import OpenAIService
from text_processor import TextProcessor

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
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', '0.3'))
        self.max_results = int(os.getenv('MAX_RESULTS', '5'))
        
    def process_question(self, question: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Processa uma pergunta usando RAG
        
        Args:
            question: Pergunta do usuário
            filters: Filtros para busca (subject, tutor, className)
            
        Returns:
            Dict com resposta, contexto usado, fontes e tokens utilizados
        """
        try:
            # 1. Buscar documentos relevantes no banco
            logger.info("Buscando documentos relevantes...")
            relevant_docs = self.db_service.search_documents(question, filters, self.max_results)
            
            if not relevant_docs:
                logger.warning("Nenhum documento relevante encontrado")
                return {
                    'answer': "Desculpe, não encontrei informações relevantes para responder sua pergunta.",
                    'context_used': "",
                    'sources': [],
                    'tokens_used': 0
                }
            
            # 2. Processar e preparar contexto
            logger.info(f"Processando {len(relevant_docs)} documentos...")
            context = self._prepare_context(relevant_docs)
            
            # 3. Gerar prompt
            prompt = self._generate_prompt(question, context)
            
            # 4. Enviar para OpenAI e obter resposta
            logger.info("Enviando para OpenAI...")
            openai_response = self.openai_service.generate_response(prompt)
            
            # 5. Preparar fontes
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
