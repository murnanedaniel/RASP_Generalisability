"""
Transformer model implementation for RASP generalization experiments.

This module implements a decoder-only Transformer specifically designed for
the counting task. The architecture emphasizes position-based attention patterns
that are crucial for length generalization according to the RASP-Generalization
Conjecture.

Key components:
- Sinusoidal positional encoding (enables position-based attention)
- Multi-head attention (learns attention patterns)  
- Feed-forward layers (local computation)
- Causal masking (autoregressive generation)

Based on the paper "What Algorithms can Transformers Learn? A Study in Length Generalization"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for enabling position-based attention patterns.
    
    This is crucial for the counting task as it allows the model to learn
    position-dependent patterns that generalize to longer sequences.
    Uses the standard sinusoidal encoding from "Attention Is All You Need".
    
    The encoding is added to token embeddings to provide position information:
    PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension (must match token embedding dimension)
            max_seq_length: Maximum sequence length to support
        """
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Token embeddings [seq_len, batch_size, d_model]
            
        Returns:
            Embeddings with positional encoding added [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for learning position-based patterns.
    
    For the counting task, attention patterns should ideally be position-based
    rather than content-based. This allows the model to:
    1. Attend to the start/end numbers based on their positions
    2. Generate counting sequences using position-relative patterns
    3. Generalize these patterns to longer sequences
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len] or [seq_len, seq_len]
            
        Returns:
            Attention output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = query.size()
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        output = self.w_o(attention_output)
        return output
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, 
                                   V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Queries [batch_size, n_heads, seq_len, d_k]
            K: Keys [batch_size, n_heads, seq_len, d_k]
            V: Values [batch_size, n_heads, seq_len, d_k]
            mask: Attention mask
            
        Returns:
            Attention output [batch_size, n_heads, seq_len, d_k]
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network for local computation.
    
    This component applies the same fully connected network to each position
    separately. For the counting task, this enables local arithmetic operations
    like incrementing/decrementing numbers.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension (typically 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward transformation.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    """
    Single transformer block combining attention and feed-forward layers.
    
    Each block enables the model to:
    1. Learn attention patterns (position-based for counting)
    2. Apply local transformations (arithmetic operations)
    3. Combine information through residual connections
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply transformer block with residual connections.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask for causal generation
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only Transformer for autoregressive sequence generation.
    
    This architecture is designed to learn algorithmic patterns that generalize
    to longer sequences. Key design choices for length generalization:
    
    1. Sinusoidal positional encoding (position-based patterns)
    2. Causal masking (autoregressive generation)
    3. Layer normalization (training stability)
    4. Appropriate initialization (helps convergence)
    
    The model learns to predict the next token given previous tokens,
    which for the counting task means learning to continue counting sequences.
    """
    
    def __init__(self, vocab_size: int, max_seq_length: int, d_model: int, 
                 n_layers: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize decoder-only transformer.
        
        Args:
            vocab_size: Size of token vocabulary
            max_seq_length: Maximum sequence length to support
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads per layer
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection to vocabulary
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights following best practices
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        Initialize model weights following standard practice.
        
        Proper initialization is important for training stability and
        achieving good length generalization performance.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal (lower triangular) mask for autoregressive generation.
        
        This ensures that each position can only attend to previous positions,
        which is essential for autoregressive language modeling.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask [seq_len, seq_len]
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training or inference.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Padding mask [batch_size, seq_len] (1=valid, 0=padding)
            
        Returns:
            Logits over vocabulary [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Token embeddings with positional encoding
        # Note: we transpose for positional encoding (expects seq_len first)
        token_embeds = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        x = token_embeds.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.position_encoding(x)  # Add positional encoding
        x = x.transpose(0, 1)  # Back to [batch_size, seq_len, d_model]
        
        # Create causal mask for autoregressive generation
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, 
                 temperature: float = 1.0, eos_token_id: Optional[int] = None) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        This is the key function for testing length generalization - we give
        the model a prompt and let it generate the counting sequence.
        
        Args:
            input_ids: Initial tokens [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = no modification)
            eos_token_id: Token ID for end of sequence (stops generation)
            
        Returns:
            Generated sequence [batch_size, seq_len + num_generated]
        """
        self.eval()
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits = self.forward(generated)
                
                # Get logits for last position
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sample next token (using multinomial for temperature > 1.0)
                if temperature == 1.0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if EOS token is generated
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return generated

class CountingTransformer(nn.Module):
    """
    Wrapper for the counting task that adds loss computation.
    
    This class wraps the decoder-only transformer and adds task-specific
    functionality like loss computation for training. It provides a clean
    interface for the counting task while maintaining the core transformer
    architecture.
    """
    
    def __init__(self, config):
        """
        Initialize counting transformer from config.
        
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        
        self.transformer = DecoderOnlyTransformer(
            vocab_size=config.model.vocab_size,
            max_seq_length=config.model.max_seq_length,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            d_ff=config.model.d_ff,
            dropout=config.model.dropout
        )
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional loss computation.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Padding mask [batch_size, seq_len]
            labels: Target labels for loss computation [batch_size, seq_len]
            
        Returns:
            Tuple of (logits, loss)
            - logits: [batch_size, seq_len, vocab_size]
            - loss: Scalar tensor if labels provided, None otherwise
        """
        # Get model outputs
        logits = self.transformer(input_ids, attention_mask)
        
        loss = None
        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1))
        
        return logits, loss
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, eos_token_id: Optional[int] = None) -> torch.Tensor:
        """
        Generate new tokens (delegates to transformer).
        
        Args:
            input_ids: Initial tokens [batch_size, seq_len]  
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            eos_token_id: End of sequence token ID
            
        Returns:
            Generated sequence [batch_size, seq_len + num_generated]
        """
        return self.transformer.generate(input_ids, max_new_tokens, temperature, eos_token_id) 