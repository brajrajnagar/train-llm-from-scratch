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
        """Create config from a dictionary (parsed from YAML)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def param_count_estimate(self) -> int:
        """Estimate total parameters (useful for sanity checks)."""
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
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # gamma (scale only)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        Returns a (max_seq_len, dim//2) tensor of complex numbers
        representing the rotation for each (position, dimension_pair).
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
            xq: (batch, seq_len, n_heads, head_dim)
            xk: (batch, seq_len, n_kv_heads, head_dim)
            freqs_cis: (seq_len, head_dim//2) complex tensor
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

        (B, T, n_kv_heads, head_dim) -> (B, T, n_heads, head_dim)

        Example with n_kv_heads=4, n_rep=3:
          [K1, K2, K3, K4] -> [K1, K1, K1, K2, K2, K2, K3, K3, K3, K4, K4, K4]
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
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """

    def __init__(self, config: ModelConfig):
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
        """Count trainable parameters (excluding tied weights)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
