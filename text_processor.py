import re
import logging
from typing import List, Dict, Any
import unicodedata

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Classe para processamento e limpeza de texto
    """
    
    def __init__(self):
        # Palavras de parada em português (básicas)
        self.stop_words = {
            'a', 'ao', 'aos', 'as', 'da', 'das', 'de', 'do', 'dos', 'e', 'em', 'é', 'na', 'nas', 'no', 'nos', 'o', 'os', 
            'para', 'por', 'que', 'se', 'uma', 'um', 'uns', 'umas', 'com', 'como', 'mas', 'ou', 'quando', 'onde', 'qual', 
            'quais', 'sobre', 'todo', 'toda', 'todos', 'todas', 'ser', 'ter', 'estar', 'fazer', 'ir', 'vir', 'ver', 'dar'
        }
        
        logger.info("TextProcessor inicializado")
    
    def clean_text(self, text: str) -> str:
        """
        Limpa e normaliza o texto
        """
        if not text:
            return ""
        
        # Normalizar unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remover caracteres especiais mantendo acentos
        text = re.sub(r'[^\w\s\-.,!?;:()\[\]{}""''áéíóúâêîôûãõàèìòùäëïöüç]', ' ', text)
        
        # Normalizar espaços
        text = re.sub(r'\s+', ' ', text)
        
        # Remover espaços no início e fim
        text = text.strip()
        
        return text
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extrai palavras-chave do texto
        """
        if not text:
            return []
        
        # Limpar texto
        clean_text = self.clean_text(text).lower()
        
        # Dividir em palavras
        words = clean_text.split()
        
        # Filtrar palavras curtas e stop words
        keywords = []
        for word in words:
            if len(word) > 2 and word not in self.stop_words:
                keywords.append(word)
        
        # Contar frequência
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Ordenar por frequência
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Retornar top keywords
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def preprocess_query(self, query: str) -> str:
        """
        Pré-processa uma consulta do usuário
        """
        # Limpar e normalizar
        processed_query = self.clean_text(query)
        
        # Remover pontuação desnecessária para busca
        processed_query = re.sub(r'[?!.,:;]', '', processed_query)
        
        return processed_query.strip()
    
    def highlight_keywords(self, text: str, keywords: List[str]) -> str:
        """
        Destaca palavras-chave no texto usando markdown
        """
        if not text or not keywords:
            return text
        
        highlighted_text = text
        
        for keyword in keywords:
            # Busca case-insensitive
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            highlighted_text = pattern.sub(f'**{keyword}**', highlighted_text)
        
        return highlighted_text
    
    def truncate_text(self, text: str, max_length: int, preserve_words: bool = True) -> str:
        """
        Trunca o texto mantendo palavras completas
        """
        if not text or len(text) <= max_length:
            return text
        
        if preserve_words:
            # Encontrar o último espaço antes do limite
            truncate_pos = text.rfind(' ', 0, max_length)
            if truncate_pos == -1:
                truncate_pos = max_length
            
            return text[:truncate_pos] + "..."
        else:
            return text[:max_length] + "..."
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extrai sentenças do texto
        """
        if not text:
            return []
        
        # Dividir por pontos, exclamações e interrogações
        sentences = re.split(r'[.!?]+', text)
        
        # Limpar e filtrar sentenças vazias
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Mínimo de 10 caracteres
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similaridade simples entre dois textos baseada em palavras comuns
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalizar textos
        words1 = set(self.clean_text(text1).lower().split())
        words2 = set(self.clean_text(text2).lower().split())
        
        # Remover stop words
        words1 = {w for w in words1 if w not in self.stop_words and len(w) > 2}
        words2 = {w for w in words2 if w not in self.stop_words and len(w) > 2}
        
        if not words1 or not words2:
            return 0.0
        
        # Calcular Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def format_academic_text(self, text: str) -> str:
        """
        Formata texto para apresentação acadêmica
        """
        if not text:
            return ""
        
        # Normalizar quebras de linha
        text = re.sub(r'\n+', '\n', text)
        
        # Adicionar formatação para títulos (linhas que terminam com :)
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line.endswith(':') and len(line) > 3:
                formatted_lines.append(f"**{line}**")
            elif line:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
