import os
import logging
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class OpenAIService:
    """
    Serviço para interagir com a API da OpenAI
    """
    
    def __init__(self):
        # Configurações da OpenAI
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o')
        self.max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '1000'))
        self.temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY não encontrada no .env")
        
        # Inicializar cliente OpenAI
        self.client = OpenAI(api_key=self.api_key)
        
        logger.info(f"OpenAIService inicializado com modelo: {self.model}")
    
    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """
        Gera uma resposta usando a API da OpenAI
        
        Args:
            prompt: Prompt formatado para enviar à OpenAI
            
        Returns:
            Dict com resposta e informações de uso
        """
        try:
            logger.info(f"Enviando prompt para OpenAI (modelo: {self.model})")
            
            # Fazer a chamada para a API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um assistente acadêmico especializado em fornecer respostas precisas e bem formatadas com base em contextos educacionais."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            # Extrair resposta e informações de uso
            answer = response.choices[0].message.content.strip()
            
            # Informações sobre uso de tokens
            usage_info = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            logger.info(f"Resposta gerada com sucesso. Tokens utilizados: {usage_info['total_tokens']}")
            
            return {
                'response': answer,
                'tokens_used': usage_info,
                'model_used': self.model
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar resposta da OpenAI: {str(e)}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Estima o número de tokens em um texto
        Implementação simples - para uso mais preciso, considere usar tiktoken
        """
        # Estimativa aproximada: 1 token ≈ 4 caracteres em português
        return len(text) // 4
    
    def validate_prompt_length(self, prompt: str) -> bool:
        """
        Valida se o prompt não excede o limite de tokens
        """
        estimated_tokens = self.count_tokens(prompt)
        
        # Deixar margem para a resposta
        max_prompt_tokens = 4000 - self.max_tokens
        
        if estimated_tokens > max_prompt_tokens:
            logger.warning(f"Prompt muito longo: {estimated_tokens} tokens estimados")
            return False
        
        return True
    
    def optimize_prompt(self, prompt: str) -> str:
        """
        Otimiza o prompt se estiver muito longo
        """
        if self.validate_prompt_length(prompt):
            return prompt
        
        # Estratégia simples: truncar o contexto mantendo a pergunta
        lines = prompt.split('\n')
        
        # Encontrar onde começa o contexto
        context_start = -1
        for i, line in enumerate(lines):
            if '**Contexto útil:**' in line:
                context_start = i
                break
        
        if context_start == -1:
            return prompt  # Não conseguiu identificar a estrutura
        
        # Manter cabeçalho e pergunta, truncar contexto
        header = lines[:context_start + 1]
        
        # Encontrar onde termina o contexto
        context_end = -1
        for i in range(context_start + 1, len(lines)):
            if '**Pergunta:**' in lines[i]:
                context_end = i
                break
        
        if context_end == -1:
            return prompt  # Não conseguiu identificar a estrutura
        
        footer = lines[context_end:]
        
        # Truncar contexto mantendo estrutura
        context_lines = lines[context_start + 1:context_end]
        
        # Manter apenas as primeiras seções do contexto
        truncated_context = []
        current_length = 0
        max_context_length = 2000  # Limite para o contexto
        
        for line in context_lines:
            if current_length + len(line) > max_context_length:
                truncated_context.append("...\n**[Contexto truncado para otimização]**")
                break
            truncated_context.append(line)
            current_length += len(line)
        
        # Recompor o prompt
        optimized_prompt = '\n'.join(header + truncated_context + footer)
        
        logger.info("Prompt otimizado devido ao comprimento")
        return optimized_prompt
