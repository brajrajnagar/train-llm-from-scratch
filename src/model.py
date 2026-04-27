# =============================================================================
# model.py -- Modern LLM Architecture (Llama-style)
# =============================================================================
#
# EVOLUTION FROM learn-gpt-from-scratch:
#   Old (GPT-2 style)         ->  New (Llama style)
#   -----------------------------------------------
#   Learned pos embeddings    ->  RoPE (Rotary Position Embeddings)
#   LayerNorm                 ->  RMSNorm
#   GELU activation           ->  SwiGLU
#   Multi-Head Attention      ->  Grouped Query Attention (GQA)
#   Bias in linear layers     ->  No bias
#   Weight tying              ->  Weight tying (kept!)
#
# WHY THESE CHANGES?
#   Each change was battle-tested by Meta (Llama), Google (PaLM),
#   and others on models from 7B to 405B parameters.
#   They're not just "different" -- they're measurably BETTER.
#
# =============================================================================

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """
    All model hyperparameters in one place.
    Loaded from YAML config files (configs/50m.yaml, configs/160m.yaml, etc.)
    
    Example (160m.yaml):
        vocab_size=50257, d_model=768, n_heads=12, n_kv_heads=4,
        n_blocks=20, d_ff=2048, seq_length=1024
    """
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12        # Number of QUERY heads
    n_kv_heads: int = 4      # Number of KEY/VALUE heads (GQA)
    n_blocks: int = 20
    d_ff: int = 2048
    seq_length: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10000.0

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Create config from a dictionary (parsed from YAML).
        
        Example:
            d = {"vocab_size": 50257, "d_model": 768, "n_heads": 12, ...}
            config = ModelConfig.from_dict(d)
            # Returns: ModelConfig(vocab_size=50257, d_model=768, n_heads=12, ...)
        """
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def param_count_estimate(self) -> int:
        """Estimate total parameters (useful for sanity checks).
        
        Example (160M model):
            vocab_size=50257, d_model=768, n_heads=12, n_kv_heads=4, n_blocks=20, d_ff=2048
            
            emb = 50257 * 768 = 38,597,376
            head_dim = 768 / 12 = 64
            attn_per_block = 768*768 + 768*4*64 + 768*4*64 + 768*768 = 1,572,864
            ffn_per_block = 3 * 768 * 2048 = 4,718,592
            norm_per_block = 2 * 768 = 1,536
            block_total = 1,572,864 + 4,718,592 + 1,536 = 6,292,992
            total = 38,597,376 + 20 * 6,292,992 + 768 = ~164M parameters
        """
        emb = self.vocab_size * self.d_model
        head_dim = self.d_model // self.n_heads
        attn_per_block = (
            self.d_model * self.d_model                        # Q
            + self.d_model * self.n_kv_heads * head_dim        # K
            + self.d_model * self.n_kv_heads * head_dim        # V
            + self.d_model * self.d_model                      # O
        )
        ffn_per_block = 3 * self.d_model * self.d_ff           # gate + up + down
        norm_per_block = 2 * self.d_model                      # 2x RMSNorm
        block_total = attn_per_block + ffn_per_block + norm_per_block
        final_norm = self.d_model
        return emb + self.n_blocks * block_total + final_norm


# =============================================================================
# RMSNorm -- Replaces LayerNorm
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    WHY RMSNorm INSTEAD OF LAYERNORM?
      LayerNorm: normalize, then shift (beta) and scale (gamma)
        -> y = gamma * (x - mean) / sqrt(var + eps) + beta

      RMSNorm: normalize by RMS only, then scale (gamma)
        -> y = gamma * x / sqrt(mean(x^2) + eps)

      Differences:
        1. No mean subtraction (no "centering")
        2. No beta (shift) parameter
        3. ~10-15% faster than LayerNorm
        4. Equally effective in practice

      Used in: Llama, Llama 2, Llama 3, Mistral, Gemma
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm.
        
        Args:
            dim: Dimension to normalize (e.g., d_model=768)
            eps: Small constant for numerical stability
            
        Example:
            norm = RMSNorm(dim=768)
            # Creates learnable weight of shape (768,)
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # gamma (scale only)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: normalize by RMS, then scale.
        
        Dimension flow:
            Input:  (batch, seq_len, d_model) e.g., (2, 1024, 768)
            Step 1: x.pow(2) -> (2, 1024, 768) - square each element
            Step 2: .mean(-1, keepdim=True) -> (2, 1024, 1) - mean over last dim
            Step 3: + eps -> (2, 1024, 1)
            Step 4: torch.rsqrt() -> (2, 1024, 1) - 1/sqrt for normalization
            Step 5: x.float() * rms -> (2, 1024, 768) - broadcast multiply
            Step 6: * self.weight -> (2, 1024, 768) - element-wise scale
            Output: (batch, seq_len, d_model) e.g., (2, 1024, 768)
        
        Example:
            x = torch.randn(2, 1024, 768)  # Random input
            norm = RMSNorm(768)
            out = norm(x)  # Output shape: (2, 1024, 768)
            # Each row is normalized to have RMS ~1, then scaled by weight
        """
        # rsqrt = 1 / sqrt(mean(x^2) + eps) -- computed in float32 for stability
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


# =============================================================================
# RoPE -- Rotary Positional Embedding
# =============================================================================

class RotaryPositionalEmbedding:
    """
    Rotary Position Embedding (RoPE) -- Su et al., 2021.

    WHY RoPE INSTEAD OF LEARNED POSITIONAL EMBEDDINGS?

      Old way (GPT-2): Learn a separate embedding for each position.
        Problem: Can't handle sequences longer than training length!
        pos_emb = nn.Embedding(max_seq_len, d_model)  # Fixed table

      New way (RoPE): Encode position by ROTATING Q and K vectors.
        Benefit 1: Works for ANY sequence length (extrapolates!)
        Benefit 2: Attention score naturally decays with distance
        Benefit 3: No extra parameters to learn

      HOW IT WORKS (simplified):
        For each pair of dimensions in Q/K, apply a 2D rotation:
          [q0, q1] -> [q0*cos(t) - q1*sin(t), q0*sin(t) + q1*cos(t)]

        The rotation angle t depends on POSITION:
          - Position 0: small rotation
          - Position 100: larger rotation
          - Position 1000: even larger rotation

        When computing Q @ K^T, the rotations SUBTRACT:
          score(pos_i, pos_j) depends on (pos_i - pos_j)
          -> Relative position is baked into the attention scores!

      Used in: Llama, Llama 2, Llama 3, Mistral, GPT-NeoX, CodeLlama
    """

    @staticmethod
    def precompute_freqs_cis(
        dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Precompute the complex exponentials for RoPE.

        Args:
            dim: Dimension of head (must be even, e.g., head_dim=64)
            max_seq_len: Maximum sequence length to support (e.g., 2048)
            theta: Base frequency (default 10000.0)
            device: Device to compute on
            
        Returns:
            Complex tensor of shape (max_seq_len, dim//2)
            Each element represents rotation angle for (position, dim_pair)
        
        Dimension flow (dim=64, max_seq_len=2048):
            Step 1: torch.arange(0, 64, 2) -> [0, 2, 4, ..., 62] (32 values)
            Step 2: / dim -> [0/64, 2/64, ..., 62/64] (normalized indices)
            Step 3: theta ** (-indices) -> frequencies [1.0, 0.75, ..., 0.01]
            Step 4: torch.arange(2048) -> positions [0, 1, ..., 2047]
            Step 5: torch.outer(positions, freqs) -> (2048, 32) angles
            Step 6: torch.polar(ones, angles) -> (2048, 32) complex rotations
            Output: (max_seq_len, dim//2) complex tensor e.g., (2048, 32)
        
        Example:
            freqs_cis = RotaryPositionalEmbedding.precompute_freqs_cis(
                dim=64, max_seq_len=2048, device="cuda"
            )
            # freqs_cis.shape = (2048, 32) complex tensor
            # freqs_cis[0, :] = rotation for position 0
            # freqs_cis[100, :] = rotation for position 100
        """
        # Frequency for each dimension pair: theta^(-2i/dim)
        # Lower dimensions rotate slowly (capture long-range patterns),
        # higher dimensions rotate fast (capture local patterns).
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        t = torch.arange(max_seq_len, device=device).float()

        # Outer product: (max_seq_len, dim//2) -- angle for each (position, dim_pair)
        freqs = torch.outer(t, freqs)

        # Convert to complex exponentials: e^(i*theta) = cos(theta) + i*sin(theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> tuple:
        """
        Apply rotary embeddings to Q and K tensors.

        Args:
            xq: Query tensor (batch, seq_len, n_heads, head_dim)
            xk: Key tensor (batch, seq_len, n_kv_heads, head_dim)
            freqs_cis: Precomputed rotations (seq_len, head_dim//2) complex
            
        Returns:
            Tuple of rotated (xq, xk) with same shapes as inputs
        
        Dimension flow (batch=2, seq_len=1024, n_heads=12, n_kv_heads=4, head_dim=64):
            Input xq: (2, 1024, 12, 64)
            Input xk: (2, 1024, 4, 64)
            freqs_cis: (1024, 32) complex
            
            Step 1: xq.reshape(2, 1024, 12, 32, 2) -> group dim pairs
            Step 2: view_as_complex -> (2, 1024, 12, 32) complex
            Step 3: freqs_cis.unsqueeze(0).unsqueeze(2) -> (1, 1024, 1, 32)
            Step 4: xq_complex * freqs_cis -> (2, 1024, 12, 32) rotated
            Step 5: view_as_real().flatten(-2) -> (2, 1024, 12, 64)
            Output xq: (2, 1024, 12, 64) - same shape, rotated
            Output xk: (2, 1024, 4, 64) - same shape, rotated
        
        Example:
            xq = torch.randn(2, 1024, 12, 64)  # Query
            xk = torch.randn(2, 1024, 4, 64)   # Key
            freqs_cis = RotaryPositionalEmbedding.precompute_freqs_cis(64, 1024)
            xq_rot, xk_rot = RotaryPositionalEmbedding.apply_rotary_emb(xq, xk, freqs_cis)
            # xq_rot.shape = (2, 1024, 12, 64) - rotated by position
            # xk_rot.shape = (2, 1024, 4, 64) - rotated by position
        """
        # Reshape to complex: group pairs of dims -> complex numbers
        # (B, T, H, D) -> (B, T, H, D//2, 2) -> complex (B, T, H, D//2)
        xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

        # Reshape freqs for broadcasting: (T, D//2) -> (1, T, 1, D//2)
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

        # Apply rotation via complex multiplication:
        #   (a + bi) * (cos(t) + i*sin(t)) = rotation by angle t
        xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
        xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)

        return xq_out.type_as(xq), xk_out.type_as(xk)


# =============================================================================
# Grouped Query Attention (GQA) -- Replaces Multi-Head Attention
# =============================================================================

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) -- Ainslie et al., 2023.

    WHY GQA INSTEAD OF STANDARD MULTI-HEAD ATTENTION (MHA)?

      In standard MHA, every query head has its OWN key and value head.
      In GQA, multiple query heads SHARE key/value heads.

      MHA (12 heads):   Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10 Q11 Q12
                        K1 K2 K3 K4 K5 K6 K7 K8 K9 K10 K11 K12  <- 12 KV heads
                        V1 V2 V3 V4 V5 V6 V7 V8 V9 V10 V11 V12

      GQA (12Q, 4KV):  Q1 Q2 Q3 | Q4 Q5 Q6 | Q7 Q8 Q9 | Q10 Q11 Q12
                           K1    |    K2     |    K3     |     K4       <- 4 KV heads
                           V1    |    V2     |    V3     |     V4

      Benefits:
        1. ~30% less memory for KV cache during inference
        2. Faster inference (less memory bandwidth)
        3. Quality is nearly identical to full MHA
        4. Training speed is similar

      Special cases:
        n_kv_heads = n_heads  ->  Standard MHA (no sharing)
        n_kv_heads = 1        ->  Multi-Query Attention (MQA, maximum sharing)

      Used in: Llama 2 (70B), Llama 3, Mistral, Gemma
    """

    def __init__(self, config: ModelConfig):
        """Initialize Grouped Query Attention.
        
        Args:
            config: ModelConfig with n_heads=12, n_kv_heads=4, d_model=768
            
        Example (160M model):
            config = ModelConfig(d_model=768, n_heads=12, n_kv_heads=4)
            attn = GroupedQueryAttention(config)
            
            # Internal shapes:
            #   head_dim = 768 / 12 = 64
            #   n_rep = 12 / 4 = 3 (each KV head serves 3 query heads)
            #   W_q: (768, 768) - Q projection (12 heads * 64)
            #   W_k: (768, 256) - K projection (4 heads * 64)
            #   W_v: (768, 256) - V projection (4 heads * 64)
            #   W_o: (768, 768) - Output projection
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads  # Query heads per KV group

        assert config.d_model % config.n_heads == 0, \
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
        assert config.n_heads % config.n_kv_heads == 0, \
            f"n_heads ({config.n_heads}) must be divisible by n_kv_heads ({config.n_kv_heads})"

        # Q projection: full size (all query heads)
        self.W_q = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        # K, V projections: reduced size (only n_kv_heads)
        self.W_k = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        # Output projection
        self.W_o = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat KV heads to match number of query heads.

        Args:
            x: KV tensor (batch, seq_len, n_kv_heads, head_dim)
            n_rep: Number of times to repeat each head
            
        Returns:
            Repeated tensor (batch, seq_len, n_heads, head_dim)
        
        Dimension flow (n_kv_heads=4, n_rep=3):
            Input:  (2, 1024, 4, 64) - 4 KV heads
            Step 1: unsqueeze(3) -> (2, 1024, 4, 1, 64) - add repeat dim
            Step 2: expand(2, 1024, 4, 3, 64) - repeat 3 times
            Step 3: reshape -> (2, 1024, 12, 64) - flatten to 12 heads
            Output: (2, 1024, 12, 64) - now matches 12 query heads
        
        Example:
            k = torch.randn(2, 1024, 4, 64)  # 4 KV heads
            k_repeated = GroupedQueryAttention.repeat_kv(k, n_rep=3)
            # k_repeated.shape = (2, 1024, 12, 64)
            # [K1, K2, K3, K4] -> [K1,K1,K1, K2,K2,K2, K3,K3,K3, K4,K4,K4]
        """
        if n_rep == 1:
            return x
        B, T, n_kv_heads, head_dim = x.shape
        x = x.unsqueeze(3).expand(B, T, n_kv_heads, n_rep, head_dim)
        return x.reshape(B, T, n_kv_heads * n_rep, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for Grouped Query Attention.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            freqs_cis: RoPE frequencies (seq_len, head_dim//2) complex
            mask: Optional attention mask (seq_len, seq_len)
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        
        Dimension flow (batch=2, seq_len=1024, d_model=768, n_heads=12, n_kv_heads=4, head_dim=64):
            Input x: (2, 1024, 768)
            
            === Project to Q, K, V ===
            Step 1: W_q(x) -> (2, 1024, 768) then .view -> (2, 1024, 12, 64)
            Step 2: W_k(x) -> (2, 1024, 256) then .view -> (2, 1024, 4, 64)
            Step 3: W_v(x) -> (2, 1024, 256) then .view -> (2, 1024, 4, 64)
            
            === Apply RoPE ===
            Step 4: apply_rotary_emb(q, k, freqs_cis)
                q: (2, 1024, 12, 64) -> rotated -> (2, 1024, 12, 64)
                k: (2, 1024, 4, 64) -> rotated -> (2, 1024, 4, 64)
            
            === Repeat KV to match Q heads ===
            Step 5: repeat_kv(k, n_rep=3) -> (2, 1024, 12, 64)
            Step 6: repeat_kv(v, n_rep=3) -> (2, 1024, 12, 64)
            
            === Transpose for attention ===
            Step 7: q.transpose(1, 2) -> (2, 12, 1024, 64)
            Step 8: k.transpose(1, 2) -> (2, 12, 1024, 64)
            Step 9: v.transpose(1, 2) -> (2, 12, 1024, 64)
            
            === Scaled Dot-Product Attention ===
            Step 10: scores = (q @ k.transpose(-2, -1)) * scale
                     (2, 12, 1024, 64) @ (2, 12, 64, 1024) -> (2, 12, 1024, 1024)
            Step 11: scores + causal_mask -> (2, 12, 1024, 1024) masked
            Step 12: attn = softmax(scores, dim=-1) -> (2, 12, 1024, 1024)
            Step 13: out = attn @ v
                     (2, 12, 1024, 1024) @ (2, 12, 1024, 64) -> (2, 12, 1024, 64)
            
            === Output projection ===
            Step 14: out.transpose(1, 2) -> (2, 1024, 12, 64)
            Step 15: .contiguous().view -> (2, 1024, 768)
            Step 16: W_o(out) -> (2, 1024, 768)
            Output: (2, 1024, 768)
        
        Example:
            x = torch.randn(2, 1024, 768)  # Input
            freqs_cis = RotaryPositionalEmbedding.precompute_freqs_cis(64, 1024)
            attn = GroupedQueryAttention(config)
            out = attn(x, freqs_cis)
            # out.shape = (2, 1024, 768)
        """
        B, T, _ = x.shape

        # Project to Q, K, V
        q = self.W_q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.W_k(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.W_v(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE to Q and K (not V -- positions don't affect "content")
        q, k = RotaryPositionalEmbedding.apply_rotary_emb(q, k, freqs_cis)

        # Repeat KV heads to match Q heads for attention computation
        k = self.repeat_kv(k, self.n_rep)
        v = self.repeat_kv(v, self.n_rep)

        # Transpose for attention: (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch's native scaled_dot_product_attention
        # This automatically uses Flash Attention when available (PyTorch 2.0+)
        # Falls back to memory-efficient attention or manual attention otherwise
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=(mask is None),  # Built-in causal mask when no explicit mask
            )
        else:
            # Manual attention for older PyTorch versions
            scale = self.head_dim ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            else:
                causal = torch.triu(
                    torch.full((T, T), float("-inf"), device=x.device), diagonal=1
                )
                scores = scores + causal
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            out = attn @ v

        # Concatenate heads and project output
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)


# =============================================================================
# SwiGLU Feed-Forward Network -- Replaces GELU FFN
# =============================================================================

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network -- Shazeer, 2020.

    WHY SwiGLU INSTEAD OF GELU?

      Old FFN (GPT-2):
        x -> Linear(d_model, 4*d_model) -> GELU -> Linear(4*d_model, d_model)
        2 matrices, expansion factor 4x

      New FFN (SwiGLU):
        gate = Linear(d_model, d_ff)    <- "how much to let through"
        up   = Linear(d_model, d_ff)    <- "what to let through"
        down = Linear(d_ff, d_model)    <- "compress back"
        output = down(SiLU(gate) * up)

      Key differences:
        1. THREE matrices instead of two (but d_ff = 8/3 * d_model, not 4x)
        2. GATING mechanism: SiLU(gate) * up = element-wise gating
        3. Total FLOPs roughly similar to old 4x expansion
        4. Empirically better training efficiency (~1-3% improvement)

      Think of it as: the gate decides WHICH features matter,
      and the up projection provides the actual features.

      Used in: Llama, Llama 2, Llama 3, PaLM, Gemma, Mistral
    """

    def __init__(self, config: ModelConfig):
        """Initialize SwiGLU FFN.
        
        Args:
            config: ModelConfig with d_model=768, d_ff=2048
            
        Example (160M model):
            config = ModelConfig(d_model=768, d_ff=2048)
            ffn = SwiGLUFFN(config)
            
            # Internal shapes:
            #   gate: (768, 2048) - "how much to let through"
            #   up:   (768, 2048) - "what to let through"
            #   down: (2048, 768) - "compress back"
            # Total params: 768*2048*3 = 4,718,592
        """
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SwiGLU FFN.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        
        Dimension flow (batch=2, seq_len=1024, d_model=768, d_ff=2048):
            Input x: (2, 1024, 768)
            
            Step 1: gate(x) = Linear(x) -> (2, 1024, 2048)
            Step 2: up(x) = Linear(x) -> (2, 1024, 2048)
            Step 3: F.silu(gate) -> (2, 1024, 2048) - SiLU activation
            Step 4: F.silu(gate) * up -> (2, 1024, 2048) - element-wise multiply
            Step 5: down(result) -> (2, 1024, 768)
            Output: (2, 1024, 768)
        
        Example:
            x = torch.randn(2, 1024, 768)
            ffn = SwiGLUFFN(config)
            out = ffn(x)
            # out.shape = (2, 1024, 768)
            
            # What happens internally:
            # gate_out = x @ gate.weight  # (2, 1024, 2048)
            # up_out = x @ up.weight      # (2, 1024, 2048)
            # activated = SiLU(gate_out) * up_out  # (2, 1024, 2048)
            # out = activated @ down.weight  # (2, 1024, 768)
        """
        # SwiGLU: SiLU(gate(x)) * up(x), then project back down
        # SiLU(x) = x * sigmoid(x) -- smooth, non-monotonic activation
        return self.down(F.silu(self.gate(x)) * self.up(x))


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Single transformer block: RMSNorm -> GQA -> Residual -> RMSNorm -> SwiGLU -> Residual.

    ARCHITECTURE:
      Input
        |
      +----------------------------+
      |  RMSNorm                   |
      |  Grouped Query Attention   |
      |  + Residual                |  <- x = x + GQA(RMSNorm(x))
      +----------------------------+
        |
      +----------------------------+
      |  RMSNorm                   |
      |  SwiGLU FFN                |
      |  + Residual                |  <- x = x + SwiGLU(RMSNorm(x))
      +----------------------------+
        |
      Output
    
    This is called "pre-norm" architecture - normalization BEFORE each sub-layer.
    More stable training than post-norm, especially for deep networks.
    """

    def __init__(self, config: ModelConfig):
        """Initialize Transformer Block.
        
        Args:
            config: ModelConfig with all hyperparameters
            
        Example (160M model):
            block = TransformerBlock(config)
            # Contains:
            #   attention_norm: RMSNorm(768)
            #   attention: GroupedQueryAttention (12 heads, 4 KV heads)
            #   ffn_norm: RMSNorm(768)
            #   ffn: SwiGLUFFN (768 -> 2048 -> 768)
        """
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for Transformer Block.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            freqs_cis: RoPE frequencies (seq_len, head_dim//2) complex
            mask: Optional attention mask (seq_len, seq_len)
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        
        Dimension flow (batch=2, seq_len=1024, d_model=768):
            Input x: (2, 1024, 768)
            
            === Attention sub-layer ===
            Step 1: self.attention_norm(x) -> (2, 1024, 768) - normalized
            Step 2: self.attention(norm_x, freqs_cis, mask) -> (2, 1024, 768)
            Step 3: x + attention_out -> (2, 1024, 768) - residual connection
                    (Now x contains original + attention output)
            
            === FFN sub-layer ===
            Step 4: self.ffn_norm(x) -> (2, 1024, 768) - normalized
            Step 5: self.ffn(norm_x) -> (2, 1024, 768)
            Step 6: x + ffn_out -> (2, 1024, 768) - residual connection
            Output: (2, 1024, 768)
        
        Example:
            x = torch.randn(2, 1024, 768)
            freqs_cis = RotaryPositionalEmbedding.precompute_freqs_cis(64, 1024)
            block = TransformerBlock(config)
            out = block(x, freqs_cis)
            # out.shape = (2, 1024, 768)
            # out = x + FFN(RMSNorm(x + GQA(RMSNorm(x))))
        """
        # Pre-norm architecture: normalize BEFORE each sub-layer
        # (More stable training than post-norm, especially for deep networks)
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# =============================================================================
# Complete LLM Model
# =============================================================================

class LLM(nn.Module):
    """
    Complete LLM model (Llama-style architecture).

    FULL ARCHITECTURE:

      Token IDs: [2980, 318, 257, ...]     ("This is a...")
        |
      +--------------------------------+
      |  Token Embedding               |  Each token -> d_model-dim vector
      |  (NO position embedding --     |
      |   RoPE handles position!)      |
      +--------------------------------+
        |
      +--------------------------------+
      |  Transformer Block 1           |
      |  Transformer Block 2           |
      |  ...                           |
      |  Transformer Block N           |  <- Each block: RMSNorm -> GQA -> SwiGLU
      +--------------------------------+
        |
      +--------------------------------+
      |  Final RMSNorm                 |
      +--------------------------------+
        |
      +--------------------------------+
      |  Output Projection (lm_head)   |  d_model -> vocab_size
      |  (tied with token embedding)   |
      +--------------------------------+
        |
      Logits -> next token probabilities
    """

    def __init__(self, config: ModelConfig):
        """Initialize LLM model.
        
        Args:
            config: ModelConfig with all hyperparameters
            
        Example (160M model):
            config = ModelConfig(
                vocab_size=50257, d_model=768, n_heads=12, n_kv_heads=4,
                n_blocks=20, d_ff=2048, seq_length=1024
            )
            model = LLM(config)
            
            # Internal structure:
            #   token_emb: (50257, 768) - token embeddings
            #   blocks: 20x TransformerBlock
            #   norm: RMSNorm(768)
            #   lm_head: (768, 50257) - output projection (tied with token_emb)
            #   freqs_cis: (2048, 32) complex - precomputed RoPE
            # Total: ~160M parameters
        """
        super().__init__()
        self.config = config

        # Token embedding (no position embedding -- RoPE handles it)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_blocks)
        ])

        # Final normalization
        self.norm = RMSNorm(config.d_model)

        # Output projection (tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying

        # Precompute RoPE frequencies (register as buffer so it moves with model)
        head_dim = config.d_model // config.n_heads
        self.register_buffer(
            "freqs_cis",
            RotaryPositionalEmbedding.precompute_freqs_cis(
                head_dim, config.seq_length * 2, config.rope_theta
            ),
            persistent=False,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights following GPT-2 / Llama conventions.

        Most weights: normal(0, 0.02)
        Residual projections (W_o and down): scaled by 1/sqrt(2*n_blocks)
          -> Prevents signal growth through deep residual streams
        
        Example (160M model, n_blocks=20):
            scale = 1 / sqrt(2 * 20) = 1 / sqrt(40) = 0.158
            W_o weights: normal(0, 0.02 * 0.158) = normal(0, 0.00316)
            down weights: normal(0, 0.02 * 0.158) = normal(0, 0.00316)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale residual projections
        scale = (2 * self.config.n_blocks) ** -0.5
        for block in self.blocks:
            torch.nn.init.normal_(block.attention.W_o.weight, mean=0.0, std=0.02 * scale)
            torch.nn.init.normal_(block.ffn.down.weight, mean=0.0, std=0.02 * scale)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> tuple:
        """
        Forward pass: token IDs -> logits (and optionally loss).

        Args:
            idx:     (batch, seq_len) token IDs
            targets: (batch, seq_len) target token IDs (optional, for training)
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss:   scalar cross-entropy loss (None if no targets)
        
        Dimension flow (batch=2, seq_len=1024, vocab_size=50257, d_model=768):
            Input idx: (2, 1024) - token IDs
            Input targets: (2, 1024) - target token IDs (optional)
            
            Step 1: self.token_emb(idx) -> (2, 1024, 768) - embed tokens
            Step 2: self.freqs_cis[:1024] -> (1024, 32) - slice RoPE freqs
            
            Step 3: For each of 20 blocks:
                block(x, freqs_cis) -> (2, 1024, 768)
                (Each block: RMSNorm -> GQA -> Residual -> RMSNorm -> SwiGLU -> Residual)
            
            Step 4: self.norm(x) -> (2, 1024, 768) - final normalization
            Step 5: self.lm_head(x) -> (2, 1024, 50257) - project to vocab
            Output logits: (2, 1024, 50257)
            
            If targets provided:
            Step 6: logits.view(-1, 50257) -> (2048, 50257)
            Step 7: targets.view(-1) -> (2048,)
            Step 8: cross_entropy(logits, targets, ignore_index=-1) -> scalar
            Output loss: scalar (e.g., 4.523)
        
        Example:
            model = LLM(config)
            idx = torch.randint(0, 50257, (2, 1024))  # Random tokens
            targets = torch.randint(0, 50257, (2, 1024))  # Target tokens
            
            logits, loss = model(idx, targets)
            # logits.shape = (2, 1024, 50257)
            # loss = scalar (e.g., 4.523)
            
            # Inference (no targets):
            logits, _ = model(idx)
            # logits.shape = (2, 1024, 50257)
            # loss = None
        """
        B, T = idx.shape

        # Embed tokens (no position embedding needed -- RoPE is applied in attention)
        x = self.token_emb(idx)

        # Get RoPE frequencies for this sequence length
        freqs_cis = self.freqs_cis[:T]

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, freqs_cis)

        # Final normalization and output projection
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,  # Supports masked labels for fine-tuning
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with temperature, top-k, and top-p.

        SAMPLING STRATEGIES (applied in order):
          1. Temperature: scale logits (lower = more focused, higher = more random)
          2. Top-k: keep only the k highest-probability tokens
          3. Top-p (nucleus): keep smallest set of tokens with cumulative prob >= p
        
        Args:
            idx: (batch, seq_len) input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.7 = balanced, 1.0 = default)
            top_k: Top-k filtering (50 = keep top 50 tokens)
            top_p: Top-p/nucleus filtering (0.9 = keep tokens covering 90% prob)
            
        Returns:
            Generated token IDs (batch, seq_len + max_new_tokens)
        
        Generation loop (batch=1, prompt_len=10, max_new_tokens=100):
            Input idx: (1, 10) - "The meaning of life"
            
            For each of 100 iterations:
                Step 1: idx_cond = idx[:, -1024:] -> (1, 10) - crop to seq_length
                Step 2: logits, _ = model(idx_cond) -> (1, 10, 50257)
                Step 3: logits = logits[:, -1, :] -> (1, 50257) - last position only
                Step 4: logits = logits / 0.7 -> (1, 50257) - temperature scaling
                Step 5: topk filtering: keep top 50, zero rest -> (1, 50257)
                Step 6: topp filtering: keep 90% cumulative prob -> (1, 50257)
                Step 7: probs = softmax(logits) -> (1, 50257) - probabilities
                Step 8: idx_next = multinomial(probs) -> (1, 1) - sample token
                Step 9: idx = cat([idx, idx_next], dim=1) -> (1, 11) growing...
            
            After 100 iterations: idx = (1, 110)
            Output: (1, 110) - original prompt + 100 new tokens
        
        Example:
            model = LLM(config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            # Encode prompt
            prompt = "The meaning of life is"
            token_ids = tokenizer.encode(prompt)  # [464, 318, 257, 3730]
            idx = torch.tensor([token_ids])  # (1, 4)
            
            # Generate
            output = model.generate(
                idx,
                max_new_tokens=100,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
            )
            # output.shape = (1, 104) - 4 prompt + 100 new tokens
            
            # Decode
            response = tokenizer.decode(output[0].tolist())
            # "The meaning of life is a philosophical question that..."
        """
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = idx[:, -self.config.seq_length:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Only last position matters

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering: zero out everything except top k tokens
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering: keep tokens until cumulative prob >= p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens above the threshold
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                # Scatter back to original indexing
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def resize_token_embeddings(self, new_vocab_size: int):
        """
        Resize token embedding and lm_head for new vocabulary size.
        Used when adding special tokens for fine-tuning.

        New token embeddings are initialized with small random values.
        Weight tying is maintained.
        
        Args:
            new_vocab_size: New vocabulary size (e.g., 50261 for +4 chat tokens)
        
        Example (fine-tuning with chat tokens):
            model = LLM(config)  # vocab_size=50257
            # Load pretrained weights
            model.load_state_dict(pretrained_checkpoint)
            
            # Resize for chat tokens
            model.resize_token_embeddings(50261)
            # token_emb: (50257, 768) -> (50261, 768) - new tokens initialized
            # lm_head: (768, 50257) -> (768, 50261) - tied with token_emb
            
            # Now model can handle <|user|>, <|assistant|>, etc.
        """
        old_vocab_size = self.config.vocab_size
        if new_vocab_size == old_vocab_size:
            return

        # Create new embedding with larger vocab
        old_emb = self.token_emb
        new_emb = nn.Embedding(new_vocab_size, self.config.d_model)
        new_emb.weight.data[:old_vocab_size] = old_emb.weight.data
        nn.init.normal_(new_emb.weight.data[old_vocab_size:], mean=0.0, std=0.02)

        self.token_emb = new_emb
        self.config.vocab_size = new_vocab_size

        # Maintain weight tying
        self.lm_head = nn.Linear(self.config.d_model, new_vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def param_count(self) -> int:
        """Count trainable parameters (excluding tied weights).
        
        Example:
            model = LLM(config)
            params = model.param_count()
            # For 160M config: ~160,000,000
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)