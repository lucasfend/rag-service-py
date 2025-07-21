from flask import Flask, request, jsonify
from rag_system import RAGSystem
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
rag_system = RAGSystem()

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Recebe JSON { "request": "pergunta do usuário" }
    Retorna JSON { "response": "resposta gerada" }
    """
    try:
        data = request.get_json()
        if not data or 'request' not in data:
            return jsonify({'response': 'Erro: campo "request" não encontrado.'}), 400
        
        question = data['request']

        logger.info(f"Pergunta recebida: {question}")

        # Processa a pergunta, busca documentos sem filtros (filters=None)
        result = rag_system.process_question(question, filters=None)

        # Retorna só a resposta no campo "response"
        return jsonify({'response': result['answer']})

    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {str(e)}")
        return jsonify({'response': f'Erro interno: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'RAG System',
        'version': '1.0.0'
    })

# No main.py, adicione uma rota de teste:
@app.route('/test-db', methods=['GET'])
def test_database():
    try:
        count = rag_system.db_service.collection.count_documents({})
        sample = list(rag_system.db_service.collection.find({}).limit(3))
        return jsonify({
            'total_documents': count,
            'sample_documents': sample
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)

