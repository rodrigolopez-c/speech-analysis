"""
Modelo Neuronal BiLSTM sobre Embeddings Contextuales (BERT/RoBERTa/XLNet)
para Detección de Metáforas a Nivel de Palabra y Oración.
Enriquecido con características lingüísticas de concreción, abstracción y WordNet.
Inspirado en la arquitectura IlliniMet.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional


class ContextualBiLSTMMetaphorModel(nn.Module):
    """
    Red Neuronal Recurrente Bidireccional (BiLSTM) alimentada por:
    1. Embeddings contextuales de Transformer (768 dim).
    2. Características lingüísticas densas de concreción/WordNet (k dim).
    Clasifica secuencialmente cada token como literal (0) o metafórico (1),
    además de producir un puntaje global para la oración.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        linguistic_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2
    ):
        super(ContextualBiLSTMMetaphorModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.linguistic_dim = linguistic_dim
        self.hidden_dim = hidden_dim
        self.total_input_dim = embedding_dim + linguistic_dim

        # Capa de proyección inicial y dropout
        self.input_dropout = nn.Dropout(dropout)

        # Red recurrente bidireccional
        self.bilstm = nn.LSTM(
            input_size=self.total_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Cabeza de clasificación a nivel de token (Token-level metaphor detection)
        # Salida de BiLSTM es hidden_dim * 2 (forward + backward)
        self.token_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # [literal_logit, metaphor_logit]
        )

        # Cabeza de clasificación a nivel de oración (Sentence-level pooling)
        self.sentence_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # logit de presencia metafórica
        )

    def forward(
        self,
        contextual_embeddings: torch.Tensor,
        linguistic_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        :param contextual_embeddings: Tensor de forma (batch_size, seq_len, embedding_dim)
        :param linguistic_features: Tensor de forma (batch_size, seq_len, linguistic_dim)
        :return: (token_logits, sentence_logits)
            token_logits: (batch_size, seq_len, 2)
            sentence_logits: (batch_size, 1)
        """
        batch_size, seq_len, _ = contextual_embeddings.shape

        if linguistic_features is not None:
            combined_input = torch.cat([contextual_embeddings, linguistic_features], dim=-1)
        else:
            # Crear características lingüísticas dummy si no se proporcionan
            zeros = torch.zeros(batch_size, seq_len, self.linguistic_dim, device=contextual_embeddings.device)
            combined_input = torch.cat([contextual_embeddings, zeros], dim=-1)

        combined_input = self.input_dropout(combined_input)

        # Paso por la BiLSTM
        lstm_out, (hn, cn) = self.bilstm(combined_input)  # (batch_size, seq_len, hidden_dim * 2)

        # 1. Predicción por token
        token_logits = self.token_classifier(lstm_out)  # (batch_size, seq_len, 2)

        # 2. Predicción a nivel de oración (Max-pooling sobre la secuencia)
        pooled, _ = torch.max(lstm_out, dim=1)  # (batch_size, hidden_dim * 2)
        sentence_logits = self.sentence_classifier(pooled)  # (batch_size, 1)

        return token_logits, sentence_logits
