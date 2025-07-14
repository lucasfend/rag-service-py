from flask import Flask, request, jsonify
from rag_system import RAGSystem
import os
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()

app = Flask(__name__)

# Inicializar o sistema RAG
rag_system = RAGSystem()

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Endpoint para receber perguntas do Spring Boot e retornar respostas usando RAG
    """
    try:
        # Obter dados da requisição
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Pergunta não fornecida',
                'status': 'error'
            }), 400
        
        question = data['question']
        
        # Parâmetros opcionais para filtrar a busca
        filters = {
            'subject': data.get('subject'),
            'tutor': data.get('tutor'),
            'className': data.get('className')
        }
        
        # Remover filtros vazios
        filters = {k: v for k, v in filters.items() if v is not None and v != ''}
        
        logger.info(f"Processando pergunta: {question}")
        logger.info(f"Filtros aplicados: {filters}")
        
        # Processar pergunta usando RAG
        response = rag_system.process_question(question, filters)
        
        return jsonify({
            'answer': response['answer'],
            'context_used': response['context_used'],
            'sources': response['sources'],
            'tokens_used': response['tokens_used'],
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {str(e)}")
        return jsonify({
            'error': f'Erro interno: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint para verificar se o serviço está funcionando
    """
    return jsonify({
        'status': 'healthy',
        'service': 'RAG System',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug)
