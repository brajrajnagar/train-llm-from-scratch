# =============================================================================
# tokenizer.py -- BPE Tokenizer Wrapper
# =============================================================================
#
# EVOLUTION FROM learn-gpt-from-scratch:
#   Old: Character-level tokenizer (65 chars, "hello" -> 5 tokens)
#   New: BPE tokenizer (50257 tokens, "hello" -> 1 token!)
#
# WHY BPE (Byte-Pair Encoding)?
#   Character-level: "understanding" -> 13 tokens (one per char)
#   BPE:             "understanding" -> 1-2 tokens ("under" + "standing")
#
#   Benefits:
#     1. Much shorter sequences -> model sees more context
#     2. Captures subword meaning ("un-" = negation)
#     3. Handles any text (even unseen words via subword fallback)
#
# We use the GPT-2 tokenizer (50,257 tokens) from HuggingFace.
# This is the SAME tokenizer used by GPT-2 and many other models.
# No need to train our own -- 50K tokens is plenty for English text.
#
# =============================================================================

from transformers import AutoTokenizer


class Tokenizer:
    """
    Thin wrapper around HuggingFace's GPT-2 BPE tokenizer.

    Attributes:
        vocab_size:    50257 (GPT-2 default), or 50261 with chat tokens
        eos_token_id:  50256 (<|endoftext|>)
        pad_token_id:  50256 (same as EOS for GPT-style models)
    """

    # Chat format special tokens (added during fine-tuning)
    SPECIAL_TOKENS = {
        "system": "<|system|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
        "end_turn": "<|end_turn|>",
    }

    def __init__(self, add_chat_tokens: bool = False):
        """
        Args:
            add_chat_tokens: If True, add special chat format tokens.
                             This increases vocab_size by 4.
                             Only set True for fine-tuning and chat.
        """
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")

        if add_chat_tokens:
            # Add special tokens for chat/instruction format
            self._tokenizer.add_special_tokens({
                "additional_special_tokens": list(self.SPECIAL_TOKENS.values())
            })

        self.vocab_size = len(self._tokenizer)
        self.eos_token_id = self._tokenizer.eos_token_id
        self.pad_token_id = self._tokenizer.eos_token_id  # GPT-style: pad = EOS

    def encode(self, text: str, add_eos: bool = False) -> list:
        """Encode text to token IDs."""
        ids = self._tokenizer.encode(text)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(ids)

    def encode_chat(self, messages: list) -> tuple:
        """
        Encode a chat conversation into token IDs with a loss mask.

        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}

        Returns:
            token_ids: Complete sequence of token IDs
            loss_mask: 1 where loss should be computed (assistant turns), 0 elsewhere

        Format:
            <|user|>What is 2+2?<|end_turn|>
            <|assistant|>2+2 equals 4.<|end_turn|>
            <|endoftext|>

        WHY A LOSS MASK?
          During fine-tuning, we only want the model to learn to generate
          ASSISTANT responses. We don't want it to learn to generate user
          prompts or formatting tokens.

          loss_mask = 0:  "don't learn from this token"
          loss_mask = 1:  "learn to predict this token"
        """
        token_ids = []
        loss_mask = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Role token (e.g., <|user|> or <|assistant|>)
            role_token = self.SPECIAL_TOKENS[role]
            role_ids = self._tokenizer.encode(role_token)
            token_ids.extend(role_ids)
            loss_mask.extend([0] * len(role_ids))

            # Content
            content_ids = self._tokenizer.encode(content)
            token_ids.extend(content_ids)
            if role == "assistant":
                loss_mask.extend([1] * len(content_ids))  # Train on assistant output
            else:
                loss_mask.extend([0] * len(content_ids))  # Don't train on user input

            # End turn token
            end_ids = self._tokenizer.encode(self.SPECIAL_TOKENS["end_turn"])
            token_ids.extend(end_ids)
            if role == "assistant":
                loss_mask.extend([1] * len(end_ids))
            else:
                loss_mask.extend([0] * len(end_ids))

        # EOS token at the end
        token_ids.append(self.eos_token_id)
        loss_mask.append(0)

        return token_ids, loss_mask
