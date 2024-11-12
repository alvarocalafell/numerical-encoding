import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel
import math
from typing import Union, Tuple, Optional

from .config import EncoderConfig
from .utils import extract_numbers_with_positions, get_context_type

class ScaledDotProductAttention(nn.Module):
    """
    Implements scaled dot-product attention mechanism with numerical awareness.
    """
    def __init__(self, temperature: float, dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        
        return output, attn

class MagnitudeAwareEncoding(nn.Module):
    """
    Implements magnitude-aware encoding for numbers using learnable embeddings.
    """
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.magnitude_embedding = nn.Embedding(
            config.num_magnitude_bins,
            config.embedding_dim
        )
        self.magnitude_scale = nn.Parameter(torch.ones(config.num_magnitude_bins))
        
        # Initialize magnitude bins with better range coverage
        boundaries = torch.logspace(
            -10, 10,  # Increased range
            config.num_magnitude_bins + 1,
            base=2    # Using base-2 for better numerical stability
        )
        self.register_buffer('bin_boundaries', boundaries)
        
        # Initialize embeddings to preserve ordering
        with torch.no_grad():
            positions = torch.arange(config.num_magnitude_bins).float()
            normalized_positions = positions / (config.num_magnitude_bins - 1)
            init_embeddings = torch.zeros(config.num_magnitude_bins, config.embedding_dim)
            init_embeddings[:, 0] = normalized_positions  # Use first dimension for ordering
            self.magnitude_embedding.weight.data.copy_(init_embeddings)
    
    def forward(self, number: torch.Tensor) -> torch.Tensor:
        if number.dim() == 1:
            number = number.unsqueeze(0)
            
        # Handle signs separately
        signs = torch.sign(number)
        abs_num = torch.abs(number)
        
        # Get magnitude bin indices with smooth transitions
        log_abs = torch.log2(abs_num + 1e-10)  # Add small epsilon to avoid log(0)
        bin_idx = torch.bucketize(
            log_abs,
            torch.log2(self.bin_boundaries)
        ).clamp(0, self.config.num_magnitude_bins - 1)
        
        # Get embeddings and apply scaling
        magnitude_emb = self.magnitude_embedding(bin_idx)
        scale = self.magnitude_scale[bin_idx].unsqueeze(-1)
        
        # Apply sign information
        signed_emb = magnitude_emb * signs.unsqueeze(-1)
        
        return signed_emb * scale



class PeriodicPositionalEncoding(nn.Module):
    """
    Implements periodic positional encoding for continuous numerical values.
    """
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.periodic_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        # Initialize frequency bands with better coverage
        dim = config.embedding_dim // 2
        freqs = torch.exp(
            torch.arange(0, dim) *
            -(math.log(config.positional_encoding_base) / (dim - 1))
        )
        self.register_buffer('freqs', freqs)
        
        # Add learnable scale factor
        self.scale_factor = nn.Parameter(torch.ones(1))
    
    def forward(self, number: torch.Tensor) -> torch.Tensor:
        if number.dim() == 1:
            number = number.unsqueeze(0)
            
        batch_size = number.shape[0]
        dim = self.config.embedding_dim // 2
        
        # Scale numbers to a reasonable range
        scaled_number = number * self.scale_factor
        pos_encoding = torch.zeros(
            batch_size,
            self.config.embedding_dim,
            device=number.device
        )
        
        # Add multi-scale periodic features
        position = scaled_number.unsqueeze(-1)
        scaled_pos = position * self.freqs[:dim]
        
        pos_encoding[:, :dim] = torch.sin(scaled_pos)
        pos_encoding[:, dim:] = torch.cos(scaled_pos)
        
        # Add magnitude-aware scaling
        magnitude_scale = torch.log1p(torch.abs(number)).unsqueeze(-1)
        pos_encoding = pos_encoding * magnitude_scale
        
        return self.periodic_proj(pos_encoding)

class NumericalEncoder(nn.Module):
    """
    Main encoder class that combines numerical and textual understanding.
    """
    def __init__(self, config: EncoderConfig = EncoderConfig()):
        super().__init__()
        self.config = config
        
        # Initialize transformer components
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Add special tokens for number marking
        special_tokens = {'additional_special_tokens': ['[NUM]', '[/NUM]']}
        self.tokenizer.add_special_tokens(special_tokens)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        
        # Initialize numerical encoding components
        self.magnitude_encoder = MagnitudeAwareEncoding(config)
        self.periodic_encoder = PeriodicPositionalEncoding(config)
        
        # Initialize attention mechanism
        self.number_text_attention = ScaledDotProductAttention(
            temperature=math.sqrt(config.embedding_dim)
        )
        
        # Final projection layers
        self.final_projection = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.Dropout(config.dropout),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )
        
        # Freeze BERT parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def encode_standalone_number(self, number: torch.Tensor) -> torch.Tensor:
        """
        Encodes a standalone number using both magnitude and periodic information.
        """
        number = number.to(self.device)
        
        # Ensure number is 2D tensor [batch_size, 1]
        if number.dim() == 1:
            number = number.unsqueeze(0)
        
        # Get embeddings
        magnitude_encoding = self.magnitude_encoder(number)  # [batch_size, embedding_dim]
        periodic_encoding = self.periodic_encoder(number)    # [batch_size, embedding_dim]
        
        print(f"Magnitude encoding shape: {magnitude_encoding.shape}")
        print(f"Periodic encoding shape: {periodic_encoding.shape}")
        
        # Ensure both tensors are 2D before concatenation
        if magnitude_encoding.dim() == 3:
            magnitude_encoding = magnitude_encoding.squeeze(1)
        if periodic_encoding.dim() == 3:
            periodic_encoding = periodic_encoding.squeeze(1)
        
        # Concatenate along the feature dimension
        combined = torch.cat([magnitude_encoding, periodic_encoding], dim=-1)
        return self.final_projection(combined)

    def encode_number(self, input_data: Union[float, int, str]) -> torch.Tensor:
        """
        Main encoding function that handles both standalone numbers and text.
        """
        if isinstance(input_data, (int, float)):
            number_tensor = torch.tensor([float(input_data)], dtype=torch.float32)
            print(f"Input number tensor shape: {number_tensor.shape}")
            return self.encode_standalone_number(number_tensor)
        return self.encode_text_with_numbers(str(input_data))

    def encode_text_with_numbers(self, text: str) -> torch.Tensor:
        """
        Encodes text containing numbers using hybrid approach.
        """
        numbers_info = extract_numbers_with_positions(text)
        
        if not numbers_info:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                text_embedding = self.text_encoder(**inputs).last_hidden_state[0]
                return text_embedding.mean(dim=0).unsqueeze(0)
        
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
            
            number_tensor = torch.tensor([number], dtype=torch.float32, device=self.device)
            number_encodings.append(
                self.encode_standalone_number(number_tensor).squeeze(0)
            )
        
        # Encode marked text
        inputs = self.tokenizer(
            marked_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(**inputs).last_hidden_state[0]
        
        # Process embeddings
        number_encodings = torch.stack(number_encodings).unsqueeze(0)
        text_embeddings = text_embeddings.unsqueeze(0)
        
        # Apply attention
        attended_encodings, _ = self.number_text_attention(
            number_encodings,
            text_embeddings,
            text_embeddings
        )
        
        # Combine embeddings
        attended_mean = attended_encodings.squeeze(0).mean(dim=0).unsqueeze(0)
        text_mean = text_embeddings.squeeze(0).mean(dim=0).unsqueeze(0)
        
        combined = torch.cat([attended_mean, text_mean], dim=-1)
        return self.final_projection(combined)
    
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Computes cosine similarity between two embeddings.
        """
        emb1 = emb1.to(self.device)
        emb2 = emb2.to(self.device)
        
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)
            
        return F.cosine_similarity(emb1, emb2).item()