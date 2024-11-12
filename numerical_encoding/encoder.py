import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel
import math
from typing import Union, Tuple, Optional

from .config import EncoderConfig
from .utils import extract_numbers_with_positions, get_context_type

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temp: float, dropout: float = 0.1):
        super().__init__()
        self.temp = temp
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn = torch.matmul(q / self.temp, k.transpose(-2, -1))
        attn = self.dropout(F.softmax(attn, dim=-1))
        return torch.matmul(attn, v)

class MagnitudeAwareEncoding(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        
        # Enhanced magnitude embedding with better separation
        self.magnitude_embedding = nn.Embedding(
            config.num_magnitude_bins * 2,
            self.embedding_dim
        )
        
        # Add scale-specific embeddings
        self.scale_embedding = nn.Embedding(
            32,  # Number of different scales
            self.embedding_dim
        )
        
        # Enhanced preprocessing network
        self.preprocess_net = nn.Sequential(
            nn.Linear(4, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(config.dropout),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU()
        )
        
        # Initialize bin boundaries for better separation
        bounds = torch.cat([
            torch.tensor([-float('inf'), 0]),
            torch.logspace(-15, -10, config.num_magnitude_bins // 6),
            torch.logspace(-10, -5, config.num_magnitude_bins // 6),
            torch.logspace(-5, 0, config.num_magnitude_bins // 6),
            torch.logspace(0, 5, config.num_magnitude_bins // 6),
            torch.logspace(5, 10, config.num_magnitude_bins // 6),
            torch.logspace(10, 15, config.num_magnitude_bins // 6)
        ])
        self.register_buffer('bin_boundaries', torch.unique(bounds))
        
        # Learnable parameters
        self.magnitude_scale = nn.Parameter(torch.ones(config.num_magnitude_bins * 2))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, number: torch.Tensor) -> torch.Tensor:
        if number.dim() == 1:
            number = number.unsqueeze(0)
        
        # Basic features with safe handling
        signs = torch.sign(number)
        abs_num = torch.abs(number)
        log_abs = torch.log1p(abs_num + 1e-15)
        scale_factor = torch.floor(torch.log10(abs_num + 1e-15))
        
        # Get scale embedding indices (shifted to be positive)
        scale_indices = (scale_factor + 16).long().clamp(0, 31)
        scale_emb = self.scale_embedding(scale_indices)
        
        # Prepare features
        features = torch.cat([
            log_abs,
            signs,
            number,
            scale_factor
        ], dim=-1)
        
        # Process features
        numerical_features = self.preprocess_net(features)
        
        # Get bin indices with smooth transitions
        bin_idx = torch.bucketize(
            log_abs,
            torch.log1p(self.bin_boundaries)
        ).clamp(0, self.config.num_magnitude_bins * 2 - 1)
        
        # Magnitude embedding with scaling
        magnitude_emb = self.magnitude_embedding(bin_idx.squeeze(-1))
        scale = F.softplus(self.magnitude_scale[bin_idx.squeeze(-1)] / self.temperature).unsqueeze(-1)
        
        # Combine embeddings with scale information
        output = (magnitude_emb + numerical_features + scale_emb) * scale
        
        # Clean up any NaN values
        output = torch.where(
            torch.isnan(output),
            torch.zeros_like(output),
            output
        )
        
        return F.normalize(output, p=2, dim=-1)

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim  # 768
        
        # Adjust number of scales to ensure divisibility
        # 768 is divisible by 12, 8, 6, 4, 3, 2
        self.num_scales = 4  # Changed from 5 to 4 to make 768 divisible by (2 * num_scales = 8)
        
        # Use 4 scales instead of 5, covering same numerical range
        scales = [0.1, 1.0, 10.0, 100.0]  # Removed 1000.0, still covers wide range
        self.register_buffer('scales', torch.tensor(scales).reshape(-1, 1))
        
        # Now embedding_dim // (2 * num_scales) = 768 // 8 = 96 dimensions per scale component
        freqs = torch.exp(
            torch.linspace(
                -math.log(config.positional_encoding_base),
                math.log(config.positional_encoding_base),
                self.embedding_dim // (2 * self.num_scales)  # 96 frequencies per scale
            )
        )
        self.register_buffer('freqs', freqs)
        
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales))
        
        self.output_net = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),  # [batch, 768] -> [batch, 768]
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)   # [batch, 768] -> [batch, 768]
        )
        
    def forward(self, number: torch.Tensor) -> torch.Tensor:
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
            
        output = self.output_net(encoding)  # [batch, 768]
        return F.normalize(output, p=2, dim=-1)

class NumberAttention(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim  # 768
        
        self.num_heads = 12
        self.head_dim = self.embedding_dim // self.num_heads  # 768 // 12 = 64
        
        # 1. Add relative position bias after head configuration
        self.relative_pos_bias = nn.Parameter(torch.zeros(12, 1, 1))  # [num_heads, 1, 1]
        
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        self.combine_proj = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        self.dropout = nn.Dropout(0.1)
        # Store temperature value directly
        self.temp = math.sqrt(self.head_dim)
        self.attn = ScaledDotProductAttention(temp=self.temp)
        
    def forward(self, mag_encoding: torch.Tensor, per_encoding: torch.Tensor) -> torch.Tensor:
        batch_size = mag_encoding.shape[0]
        
        # Add sequence dimension if not present
        if mag_encoding.dim() == 2:
            mag_encoding = mag_encoding.unsqueeze(1)  # [batch, 1, 768]
        if per_encoding.dim() == 2:
            per_encoding = per_encoding.unsqueeze(1)  # [batch, 1, 768]
            
        # Multi-head processing
        q = self.q_proj(mag_encoding)  # [batch, 1, 768]
        k = self.k_proj(per_encoding)  # [batch, 1, 768]
        v = self.v_proj(per_encoding)  # [batch, 1, 768]
        
        # Reshape for attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, 12, 1, 64]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Extract original numbers from encodings (approximate from magnitude)
        numbers = torch.norm(mag_encoding, dim=-1, keepdim=True)  # [batch, 1, 1]
        relative_pos = torch.log1p(
            torch.abs(numbers.unsqueeze(-1) - numbers.unsqueeze(-2))
        )  # [batch, 1, 1]
        
        # Apply attention with relative position bias
        attn = torch.matmul(q / self.temp, k.transpose(-2, -1))  # [batch, 12, 1, 1]
        # Add relative position bias
        attn = attn + self.relative_pos_bias * relative_pos.unsqueeze(1)  # Broadcasting to all heads
        attn = self.dropout(F.softmax(attn, dim=-1))
        attn_output = torch.matmul(attn, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, 1, 12, 64]
        attn_output = attn_output.view(batch_size, -1, self.embedding_dim)  # [batch, 1, 768]
        
        # Remove sequence dimension and combine with original inputs
        combined = torch.cat([
            attn_output.squeeze(1),  # [batch, 768]
            mag_encoding.squeeze(1) + per_encoding.squeeze(1)  # [batch, 768]
        ], dim=-1)  # [batch, 1536]
        
        output = self.combine_proj(combined)  # [batch, 768]
        return F.normalize(output, p=2, dim=-1)
    
class TextNumberAttention(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim  # 768
        
        self.num_heads = 12
        self.head_dim = self.embedding_dim // self.num_heads  # 768 // 12 = 64
        
        # 1. Add context encoder before projections
        self.context_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 5)  # 5 context types
        )
        
        # 2. Add context scaling parameter
        self.context_scaling = nn.Parameter(torch.ones(5))
        
        # Existing projections
        self.q_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
        self.k_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
        self.v_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
        
        self.combine_net = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(config.dropout),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
        
        self.attention = ScaledDotProductAttention(
            temp=math.sqrt(self.head_dim),
            dropout=config.dropout
        )
        
    def forward(self, number_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        batch_size = text_emb.shape[0]
        
        # 1. Get context information first
        context_logits = self.context_encoder(text_emb.mean(dim=1))  # [batch_size, 5]
        context_weights = F.softmax(context_logits, dim=-1)  # [batch_size, 5]
        
        # Handle dimensions
        if number_emb.dim() == 2:
            number_emb = number_emb.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # Project inputs
        q = self.q_proj(number_emb)  # [batch_size, num_numbers/1, embedding_dim]
        k = self.k_proj(text_emb)    # [batch_size, seq_len, embedding_dim]
        v = self.v_proj(text_emb)    # [batch_size, seq_len, embedding_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention
        attn_output = self.attention(q, k, v)  # [batch_size, num_heads, num_numbers/1, head_dim]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.embedding_dim)
        
        # 2. Apply context-aware scaling
        context_scale = torch.matmul(context_weights, self.context_scaling)  # [batch_size, 1]
        attn_output = attn_output * context_scale.unsqueeze(-1)  # [batch_size, seq_len, embedding_dim]
        
        # Pool embeddings
        pooled_numbers = number_emb.mean(1)   # [batch_size, embedding_dim]
        pooled_context = attn_output.mean(1)  # [batch_size, embedding_dim]
        
        # Combine embeddings
        combined = torch.cat([pooled_numbers, pooled_context], dim=-1)
        output = self.combine_net(combined)
        
        return F.normalize(output, dim=-1)

class NumericalEncoder(nn.Module):
    def __init__(self, config: EncoderConfig = EncoderConfig()):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim  # Should be 768
        
        # Core components
        self.magnitude_encoder = MagnitudeAwareEncoding(config)
        self.periodic_encoder = PeriodicPositionalEncoding(config)
        self.number_attention = NumberAttention(config)
        
        # Text processing components remain the same
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        special_tokens = {'additional_special_tokens': ['[NUM]', '[/NUM]']}
        self.tokenizer.add_special_tokens(special_tokens)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        
        self.text_number_attention = TextNumberAttention(config)
        
        # Add dimension checks in final projection
        self.final_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),  # [batch, 768] -> [batch, 768]
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(config.dropout),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)  # [batch, 768] -> [batch, 768]
        )
        
        # Freeze BERT
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def encode_standalone_number(self, number: torch.Tensor) -> torch.Tensor:
        """Handle standalone number encoding with explicit dimension checks"""
        # Ensure number is on correct device
        number = number.to(self.device)
        
        if number.dim() == 1:
            number = number.unsqueeze(0)  # [1, 1]
            
        # Get encodings with dimension checks
        magnitude_encoding = self.magnitude_encoder(number)  # Should be [1, 768]
        periodic_encoding = self.periodic_encoder(number)    # Should be [1, 768]
        
        # Add dimension checks
        assert magnitude_encoding.size(-1) == self.embedding_dim, \
            f"Magnitude encoding dimension mismatch: {magnitude_encoding.size()}"
        assert periodic_encoding.size(-1) == self.embedding_dim, \
            f"Periodic encoding dimension mismatch: {periodic_encoding.size()}"
        
        # Ensure both encodings are 2D tensors [batch_size, embedding_dim]
        if magnitude_encoding.dim() == 1:
            magnitude_encoding = magnitude_encoding.unsqueeze(0)
        if periodic_encoding.dim() == 1:
            periodic_encoding = periodic_encoding.unsqueeze(0)
            
        # Process through number attention
        combined = self.number_attention(
            magnitude_encoding,  # [1, 768]
            periodic_encoding    # [1, 768]
        )  # Should output [1, 768]
        
        # Final projection with dimension check
        output = self.final_projection(combined)  # Should maintain [1, 768]
        assert output.size(-1) == self.embedding_dim, \
            f"Final output dimension mismatch: {output.size()}"
            
        return F.normalize(output, p=2, dim=-1)
    
    def encode_text_with_numbers(self, text: str) -> torch.Tensor:
        numbers_info = extract_numbers_with_positions(text)
        
        if not numbers_info:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                text_embedding = self.text_encoder(**inputs).last_hidden_state
                pooled = text_embedding.mean(dim=1)
                return F.normalize(pooled, p=2, dim=-1)
        
        # Process text with numbers
        marked_text = text
        offset = 0
        number_encodings = []
        
        for number, start, end in numbers_info:
            num_token = f'[NUM]{len(number_encodings)}[/NUM]'
            marked_text = (
                marked_text[:start + offset] +
                num_token +
                marked_text[end + offset:]
            )
            offset += len(num_token) - (end - start)
            
            # Encode number
            number_tensor = torch.tensor([float(number)], dtype=torch.float32, device=self.device)
            encoded_number = self.encode_standalone_number(number_tensor)  # [1, embedding_dim]
            number_encodings.append(encoded_number)
        
        # Process text
        inputs = self.tokenizer(
            marked_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(**inputs).last_hidden_state  # [1, seq_len, embedding_dim]
        
        if number_encodings:
            # Stack number encodings and ensure proper dimensions
            number_encodings = torch.cat(number_encodings, dim=0)  # [num_numbers, embedding_dim]
            number_encodings = number_encodings.unsqueeze(0)  # [1, num_numbers, embedding_dim]
            
            # Process through attention
            combined = self.text_number_attention(
                number_encodings,  # [1, num_numbers, embedding_dim]
                text_embeddings   # [1, seq_len, embedding_dim]
            )
            return F.normalize(combined, p=2, dim=-1)
        
        return F.normalize(text_embeddings.mean(dim=1), p=2, dim=-1)

    def encode_number(self, input_data: Union[float, int, str]) -> torch.Tensor:
        """Main encoding entry point with dimension checks"""
        try:
            if isinstance(input_data, (int, float)):
                number_tensor = torch.tensor([float(input_data)], dtype=torch.float32)
                encoded = self.encode_standalone_number(number_tensor)  # Should be [1, 768]
                assert encoded.size(-1) == self.embedding_dim, \
                    f"Encoded standalone number dimension mismatch: {encoded.size()}"
                return F.normalize(encoded, p=2, dim=-1)
            else:
                text_input = str(input_data).strip()
                if not text_input:
                    raise ValueError("Empty input string")
                    
                encoded = self.encode_text_with_numbers(text_input)  # Should be [1, 768]
                assert encoded.size(-1) == self.embedding_dim, \
                    f"Encoded text with numbers dimension mismatch: {encoded.size()}"
                return F.normalize(encoded, p=2, dim=-1)
                
        except Exception as e:
            print(f"Error encoding input {input_data}: {str(e)}")
            # Return zero vector with correct dimension
            return torch.zeros(1, self.embedding_dim, device=self.device)
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        emb1 = emb1.to(self.device)
        emb2 = emb2.to(self.device)
        
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)
            
        return F.cosine_similarity(emb1, emb2).item()
    
    