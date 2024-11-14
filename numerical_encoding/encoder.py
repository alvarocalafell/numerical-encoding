"""
Main encoder module for handling numerical data in various contexts.
Integrates magnitude-aware encoding, periodic encoding, and text-number attention
to create unified representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel
import math
from typing import Union, Tuple, List

from .config import EncoderConfig
from .utils import extract_numbers_with_positions, get_context_type



class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism with dropout.
    
    Attributes:
        temp (float): Temperature scaling factor for attention scores
        dropout (nn.Dropout): Dropout layer for regularization
    """
    
    def __init__(self, temp: float, dropout: float = 0.1):
        super().__init__()
        self.temp = temp
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled dot-product attention.
        
        Args:
            q: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
            k: Key tensor [batch_size, num_heads, seq_len_k, head_dim]
            v: Value tensor [batch_size, num_heads, seq_len_v, head_dim]
            
        Returns:
            torch.Tensor: Attention output [batch_size, num_heads, seq_len_q, head_dim]
        """
        attn = torch.matmul(q / self.temp, k.transpose(-2, -1))  # [batch, heads, seq_len_q, seq_len_k]
        attn = self.dropout(F.softmax(attn, dim=-1))  # [batch, heads, seq_len_q, seq_len_k]
        return torch.matmul(attn, v)  # [batch, heads, seq_len_q, head_dim]

class NumberAttention(nn.Module):
    """
    Specialized attention mechanism for processing numerical embeddings.
    
    Attributes:
        config (EncoderConfig): Configuration object
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
    """
    
    def __init__(self, config: 'EncoderConfig'):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim  # 768
        self.num_heads = 12
        self.head_dim = self.embedding_dim // self.num_heads  # 64
        
        # Relative position bias for capturing numerical relationships
        self.relative_pos_bias = nn.Parameter(torch.zeros(12, 1, 1))  # [heads, 1, 1]
        
        # Projection layers
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)  # [768, 768]
        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)  # [768, 768]
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)  # [768, 768]
        
        # Output projection
        self.combine_proj = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),  # [1536, 768]
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)  # [768, 768]
        )
        
        self.dropout = nn.Dropout(0.1)
        self.temp = (self.head_dim ** 0.5)
        self.attn = ScaledDotProductAttention(temp=self.temp)
    
    def forward(self, mag_encoding: torch.Tensor, per_encoding: torch.Tensor) -> torch.Tensor:
        """
        Process numerical encodings through attention mechanism.
        
        Args:
            mag_encoding: Magnitude encoding [batch_size, 768]
            per_encoding: Periodic encoding [batch_size, 768]
            
        Returns:
            torch.Tensor: Combined attention output [batch_size, 768]
        """
        batch_size = mag_encoding.shape[0]
        
        # Add sequence dimension if needed
        if mag_encoding.dim() == 2:
            mag_encoding = mag_encoding.unsqueeze(1)  # [batch, 1, 768]
        if per_encoding.dim() == 2:
            per_encoding = per_encoding.unsqueeze(1)  # [batch, 1, 768]
        
        # Project inputs
        q = self.q_proj(mag_encoding)  # [batch, 1, 768]
        k = self.k_proj(per_encoding)  # [batch, 1, 768]
        v = self.v_proj(per_encoding)  # [batch, 1, 768]
        
        # Reshape for attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, 12, 1, 64]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, 12, 1, 64]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, 12, 1, 64]
        
        # Compute numerical relationships
        numbers = torch.norm(mag_encoding, dim=-1, keepdim=True)  # [batch, 1, 1]
        relative_pos = torch.log1p(
            torch.abs(numbers.unsqueeze(-1) - numbers.unsqueeze(-2))
        )  # [batch, 1, 1]
        
        # Apply attention with relative position bias
        attn = torch.matmul(q / self.temp, k.transpose(-2, -1))  # [batch, 12, 1, 1]
        attn = attn + self.relative_pos_bias * relative_pos.unsqueeze(1)  # Add bias to all heads
        attn = self.dropout(F.softmax(attn, dim=-1))
        attn_output = torch.matmul(attn, v)  # [batch, 12, 1, 64]
        
        # Reshape and combine
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, 1, 12, 64]
        attn_output = attn_output.view(batch_size, -1, self.embedding_dim)  # [batch, 1, 768]
        
        # Combine with original inputs
        combined = torch.cat([
            attn_output.squeeze(1),  # [batch, 768]
            mag_encoding.squeeze(1) + per_encoding.squeeze(1)  # [batch, 768]
        ], dim=-1)  # [batch, 1536]
        
        output = self.combine_proj(combined)  # [batch, 768]
        return F.normalize(output, p=2, dim=-1)

class MagnitudeAwareEncoding(nn.Module):
    """
    Encoding module that preserves magnitude relationships between numbers.
    
    Combines direct magnitude representation with scale-aware features to create
    embeddings that maintain numerical relationships.
    """
    
    def __init__(self, config: 'EncoderConfig'):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim  # 768
        
        # Magnitude representation
        self.magnitude_embedding = nn.Embedding(
            config.num_magnitude_bins * 2,  # 20 bins total
            self.embedding_dim  # [20, 768]
        )
        
        # Scale embedding for handling different numerical ranges
        self.scale_embedding = nn.Sequential(
            nn.Linear(1, self.embedding_dim // 4),  # [batch, 1] -> [batch, 192]
            nn.LayerNorm(self.embedding_dim // 4),
            nn.GELU(),
            nn.Linear(self.embedding_dim // 4, self.embedding_dim)  # [batch, 192] -> [batch, 768]
        )
        
        # Feature encoders
        self.numerical_encoder = nn.Sequential(
            nn.Linear(2, self.embedding_dim),  # [batch, 2] -> [batch, 768]
            nn.LayerNorm(self.embedding_dim),
            nn.GELU()
        )
        
        self.contextual_encoder = nn.Sequential(
            nn.Linear(2, self.embedding_dim),  # [batch, 2] -> [batch, 768]
            nn.LayerNorm(self.embedding_dim),
            nn.GELU()
        )
        
        # Initialize bin boundaries
        bounds = torch.cat([
            torch.tensor([-float('inf')]),
            -torch.logspace(0, 15, config.num_magnitude_bins - 1),
            torch.tensor([0.0]),
            torch.logspace(0, 15, config.num_magnitude_bins - 1)
        ])
        self.register_buffer('bin_boundaries', torch.unique(bounds))
        
        # Learnable parameters
        self.magnitude_scale = nn.Parameter(torch.ones(config.num_magnitude_bins * 2))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Feature attention
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
    
    def forward(self, number: torch.Tensor) -> torch.Tensor:
        """
        Encode numbers with magnitude awareness.
        
        Args:
            number: Input tensor of numbers [batch_size, 1]
            
        Returns:
            torch.Tensor: Magnitude-aware embeddings [batch_size, 768]
        """
        if number.dim() == 1:
            number = number.unsqueeze(0)  # [batch, 1]
        
        # Extract numerical features
        signs = torch.sign(number)  # [batch, 1]
        abs_num = torch.abs(number)  # [batch, 1]
        log_abs = torch.log1p(abs_num)  # [batch, 1]
        scale_factor = torch.floor(torch.log10(abs_num + 1e-10))  # [batch, 1]
        
        # Calculate normalized features
        numerical_features = torch.cat([
            number / (torch.max(torch.abs(number)) + 1e-10),  # Normalized value
            log_abs / (torch.max(log_abs) + 1e-10)  # Normalized log value
        ], dim=-1)  # [batch, 2]
        
        contextual_features = torch.cat([
            signs,
            scale_factor / 16.0  # Normalized scale
        ], dim=-1)  # [batch, 2]
        
        # Process features
        num_encoding = self.numerical_encoder(numerical_features)  # [batch, 768]
        ctx_encoding = self.contextual_encoder(contextual_features)  # [batch, 768]
        
        # Get magnitude embedding
        bin_idx = torch.bucketize(
            number,
            self.bin_boundaries
        ).clamp(0, self.config.num_magnitude_bins * 2 - 1)  # [batch, 1]
        magnitude_emb = self.magnitude_embedding(bin_idx.squeeze(-1))  # [batch, 768]
        
        # Get scale embedding
        scale_emb = self.scale_embedding(scale_factor.unsqueeze(-1)).squeeze(1)  # [batch, 768]
        
        # Combine features with attention - ensure all tensors have same dimensions
        features = torch.stack([
            num_encoding,      # [batch, 768]
            magnitude_emb,     # [batch, 768]
            scale_emb,        # [batch, 768]
            ctx_encoding      # [batch, 768]
        ], dim=1)  # [batch, 4, 768]
        
        # Apply attention
        attn_output, _ = self.feature_attention(
            features,
            features,
            features
        )  # [batch, 4, 768]
        
        # Weighted pooling
        weights = torch.tensor([0.4, 0.3, 0.2, 0.1], device=number.device)
        output = (attn_output * weights.view(1, 4, 1)).sum(dim=1)  # [batch, 768]
        
        # Apply magnitude scaling
        scale = F.softplus(
            self.magnitude_scale[bin_idx.squeeze(-1)] / self.temperature
        ).unsqueeze(-1)  # [batch, 1]
        
        output = output * scale  # [batch, 768]
        
        # Clean up any NaN values
        output = torch.where(
            torch.isnan(output),
            torch.zeros_like(output),
            output
        )
        
        return F.normalize(output, p=2, dim=-1)  # [batch, 768]
    
class PeriodicPositionalEncoding(nn.Module):
    """
    Periodic positional encoding for capturing numerical patterns and relationships.
    Uses multiple frequency scales to represent numbers in a continuous space.
    
    Attributes:
        config (EncoderConfig): Configuration object
        embedding_dim (int): Output embedding dimension
        num_scales (int): Number of frequency scales
        scales (torch.Tensor): Scale factors for different frequency bands
    """
    
    def __init__(self, config: 'EncoderConfig'):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim  # 768
        
        # Use 4 scales for better division of embedding dimension
        self.num_scales = 4  # 768 / (2 * 4) = 96 dimensions per scale
        
        # Define scales for different frequency bands
        scales = [0.1, 1.0, 10.0, 100.0]  # Covers wide numerical range
        self.register_buffer('scales', torch.tensor(scales).reshape(-1, 1))
        
        # Calculate frequencies for each scale
        freqs = torch.exp(
            torch.linspace(
                -math.log(config.positional_encoding_base),
                math.log(config.positional_encoding_base),
                self.embedding_dim // (2 * self.num_scales)  # 96 frequencies per scale
            )
        )
        self.register_buffer('freqs', freqs)
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales))
        
        # Output projection
        self.output_net = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),  # [batch, 768] -> [batch, 768]
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)   # [batch, 768] -> [batch, 768]
        )
    
    def forward(self, number: torch.Tensor) -> torch.Tensor:
        """
        Compute periodic encoding for input numbers.
        
        Args:
            number: Input tensor [batch_size, 1]
            
        Returns:
            torch.Tensor: Periodic positional encoding [batch_size, 768]
        """
        if number.dim() == 1:
            number = number.unsqueeze(0)  # [batch, 1]
        
        # Scale input numbers
        scaled_numbers = number.unsqueeze(-1) * self.scales  # [batch, num_scales=4]
        
        # Calculate periodic features
        features = []
        for i in range(self.num_scales):
            scaled_pos = scaled_numbers[:, i:i+1] * self.freqs  # [batch, 1, 96]
            features.extend([
                torch.sin(scaled_pos) * self.scale_weights[i],  # [batch, 1, 96]
                torch.cos(scaled_pos) * self.scale_weights[i]   # [batch, 1, 96]
            ])
        
        # Combine features: 4 scales * 2 (sin/cos) * 96 = 768 dimensions
        encoding = torch.cat(features, dim=-1)  # [batch, 768]
        
        assert encoding.size(-1) == self.embedding_dim, \
            f"Periodic encoding dimension mismatch: {encoding.size()}"
            
        # Final projection
        output = self.output_net(encoding)  # [batch, 768]
        return F.normalize(output, p=2, dim=-1)
    
class TextNumberAttention(nn.Module):
    """
    Attention mechanism for combining numerical and textual representations.
    Includes context-aware scaling and multi-head attention.
    
    Attributes:
        config (EncoderConfig): Configuration object
        embedding_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        head_dim (int): Dimension per attention head
    """
    
    def __init__(self, config: 'EncoderConfig'):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim  # 768
        
        self.num_heads = 12
        self.head_dim = self.embedding_dim // self.num_heads  # 64
        
        # Context encoder for semantic understanding
        self.context_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),  # [batch, 768] -> [batch, 256]
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 5)  # [batch, 256] -> [batch, 5] (5 context types)
        )
        
        # Context scaling parameter
        self.context_scaling = nn.Parameter(torch.ones(5))
        
        # Projection layers with layer normalization
        self.q_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),  # [batch, 768] -> [batch, 768]
            nn.LayerNorm(self.embedding_dim)
        )
        self.k_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),  # [batch, 768] -> [batch, 768]
            nn.LayerNorm(self.embedding_dim)
        )
        self.v_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),  # [batch, 768] -> [batch, 768]
            nn.LayerNorm(self.embedding_dim)
        )
        
        # Output combination network
        self.combine_net = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),  # [batch, 1536] -> [batch, 768]
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(config.dropout),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),  # [batch, 768] -> [batch, 768]
            nn.LayerNorm(self.embedding_dim)
        )
        
        # Initialize attention mechanism
        self.attention = ScaledDotProductAttention(
            temp=math.sqrt(self.head_dim),
            dropout=config.dropout
        )
    
    def forward(self, number_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Process number and text embeddings through attention mechanism.
        
        Args:
            number_emb: Number embeddings [batch_size, num_numbers, embedding_dim]
            text_emb: Text embeddings [batch_size, seq_len, embedding_dim]
            
        Returns:
            torch.Tensor: Combined embeddings [batch_size, embedding_dim]
        """
        batch_size = text_emb.shape[0]
        
        # Get context information
        context_logits = self.context_encoder(text_emb.mean(dim=1))  # [batch, 5]
        context_weights = F.softmax(context_logits, dim=-1)  # [batch, 5]
        
        # Handle dimensions
        if number_emb.dim() == 2:
            number_emb = number_emb.unsqueeze(1)  # [batch, 1, 768]
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)  # [batch, 1, 768]
        
        # Project inputs
        q = self.q_proj(number_emb)  # [batch, num_numbers, 768]
        k = self.k_proj(text_emb)    # [batch, seq_len, 768]
        v = self.v_proj(text_emb)    # [batch, seq_len, 768]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, 12, num_numbers, 64]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, 12, seq_len, 64]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, 12, seq_len, 64]
        
        # Apply attention
        attn_output = self.attention(q, k, v)  # [batch, 12, num_numbers, 64]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, num_numbers, 12, 64]
        attn_output = attn_output.view(batch_size, -1, self.embedding_dim)  # [batch, num_numbers, 768]
        
        # Apply context-aware scaling
        context_scale = torch.matmul(context_weights, self.context_scaling)  # [batch, 1]
        attn_output = attn_output * context_scale.unsqueeze(-1)  # [batch, num_numbers, 768]
        
        # Pool embeddings
        pooled_numbers = number_emb.mean(1)   # [batch, 768]
        pooled_context = attn_output.mean(1)  # [batch, 768]
        
        # Combine embeddings
        combined = torch.cat([pooled_numbers, pooled_context], dim=-1)  # [batch, 1536]
        output = self.combine_net(combined)  # [batch, 768]
        
        return F.normalize(output, dim=-1)

class NumericalEncoder(nn.Module):
    """
    Neural encoder for numerical data that handles both standalone numbers and numbers in text.
    
    Architecture Overview:
        Standalone Numbers:
            Input Number -> Magnitude Encoding [768] || Periodic Encoding [768]
                       -> Number Attention [768]
                       -> Final Projection [768]
        
        Text with Numbers:
            Text -> BERT Encoding [768]
            Numbers -> Standalone Encoding [768]
            -> Text-Number Attention [768]
            -> Final Projection [768]
    """
    
    def __init__(self, config: EncoderConfig = EncoderConfig()):
        """
        Initialize the NumericalEncoder.
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim  # 768
        
        # Initialize core numerical encoding components
        self.magnitude_encoder = MagnitudeAwareEncoding(config)
        self.periodic_encoder = PeriodicPositionalEncoding(config)
        self.number_attention = NumberAttention(config)
        
        # Initialize text processing components
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Add special tokens for number marking
        special_tokens = {'additional_special_tokens': ['[NUM]', '[/NUM]']}
        self.tokenizer.add_special_tokens(special_tokens)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        
        # Initialize text-number attention
        self.text_number_attention = TextNumberAttention(config)
        
        # Final projection layer
        self.final_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(config.dropout),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        # Freeze BERT parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def encode_standalone_number(self, number: torch.Tensor) -> torch.Tensor:
        """
        Encode a standalone number.
        
        Args:
            number: Tensor of shape [batch_size] or [batch_size, 1]
            
        Returns:
            torch.Tensor: Encoded number [batch_size, 768]
        """
        # Ensure number is on correct device
        number = number.to(self.device)
        
        if number.dim() == 1:
            number = number.unsqueeze(-1)  # [batch, 1]
            
        try:
            # Get encodings
            magnitude_encoding = self.magnitude_encoder(number)  # [batch, 768]
            periodic_encoding = self.periodic_encoder(number)    # [batch, 768]
            
            # Process through number attention
            combined = self.number_attention(
                magnitude_encoding,
                periodic_encoding
            )  # [batch, 768]
            
            # Final projection
            output = self.final_projection(combined)  # [batch, 768]
            return F.normalize(output, p=2, dim=-1)
            
        except Exception as e:
            print(f"Error in standalone number encoding: {e}")
            return torch.zeros(1, self.embedding_dim, device=self.device)
    
    def encode_text_with_numbers(self, text: str) -> torch.Tensor:
        """
        Encode text containing numbers.
        
        Args:
            text: Input text string containing numbers
            
        Returns:
            torch.Tensor: Encoded text with numbers [batch_size, 768]
        """
        try:
            # Extract numbers and their positions
            numbers_info = extract_numbers_with_positions(text)
            
            # If no numbers found, process as regular text
            if not numbers_info:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_sequence_length
                ).to(self.device)
                
                with torch.no_grad():
                    text_embedding = self.text_encoder(**inputs).last_hidden_state  # [1, seq_len, 768]
                    pooled = text_embedding.mean(dim=1)  # [1, 768]
                    return F.normalize(pooled, p=2, dim=-1)
            
            # Process text with numbers
            marked_text = text
            offset = 0
            number_encodings = []
            
            # Replace numbers with markers and encode them
            for number, start, end in numbers_info:
                # Add markers around number
                num_token = f'[NUM]{len(number_encodings)}[/NUM]'
                marked_text = (
                    marked_text[:start + offset] +
                    num_token +
                    marked_text[end + offset:]
                )
                offset += len(num_token) - (end - start)
                
                # Encode number
                number_tensor = torch.tensor([float(number)], dtype=torch.float32, device=self.device)
                encoded_number = self.encode_standalone_number(number_tensor)  # [1, 768]
                number_encodings.append(encoded_number)
            
            # Process marked text through BERT
            inputs = self.tokenizer(
                marked_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_sequence_length
            ).to(self.device)
            
            with torch.no_grad():
                text_embeddings = self.text_encoder(**inputs).last_hidden_state  # [1, seq_len, 768]
            
            if number_encodings:
                # Combine number encodings
                number_encodings = torch.cat(number_encodings, dim=0)  # [num_numbers, 768]
                number_encodings = number_encodings.unsqueeze(0)  # [1, num_numbers, 768]
                
                # Process through text-number attention
                combined = self.text_number_attention(
                    number_encodings,
                    text_embeddings
                )  # [1, 768]
                return F.normalize(combined, p=2, dim=-1)
            
            # Fallback to mean pooling if something went wrong with number processing
            return F.normalize(text_embeddings.mean(dim=1), p=2, dim=-1)
            
        except Exception as e:
            print(f"Error in text with numbers encoding: {e}")
            return torch.zeros(1, self.embedding_dim, device=self.device)
    
    def encode_number(self, input_data: Union[float, int, str]) -> torch.Tensor:
        """
        Main encoding entry point handling both standalone numbers and text.
        
        Args:
            input_data: Number (float/int) or text containing numbers (str)
            
        Returns:
            torch.Tensor: Encoded representation [1, 768]
            
        Raises:
            ValueError: If input string is empty
            TypeError: If input type is not supported
        """
        try:
            if isinstance(input_data, (int, float)):
                number_tensor = torch.tensor([float(input_data)], dtype=torch.float32)
                return self.encode_standalone_number(number_tensor)
            elif isinstance(input_data, str):
                text_input = input_data.strip()
                if not text_input:
                    raise ValueError("Empty input string")
                return self.encode_text_with_numbers(text_input)
            else:
                raise TypeError(f"Unsupported input type: {type(input_data)}")
                
        except Exception as e:
            print(f"Error encoding input {input_data}: {str(e)}")
            return torch.zeros(1, self.embedding_dim, device=self.device)
    
    
    def batch_encode(self, inputs: List[Union[float, int, str]]) -> torch.Tensor:
        """
        Encode a batch of inputs efficiently.
        
        Args:
            inputs: List of numbers or texts to encode
            
        Returns:
            torch.Tensor: Batch of encodings [batch_size, 768]
        """
        encodings = []
        for input_data in inputs:
            encoding = self.encode_number(input_data)
            encodings.append(encoding)
        return torch.cat(encodings, dim=0)
    
    