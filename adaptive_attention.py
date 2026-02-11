
import torch
import torch.nn as nn
from einops import rearrange
from chronos.chronos2.layers import GroupSelfAttention, AttentionOutput, Chronos2CoreConfig

class RelevanceBias(nn.Module):
    """
    Computes a variable-to-variable importance bias for group attention.
    
    This module learns a scalar importance score between every pair of time series (variates)
    in the group. The result is added to the attention logits before softmax.
    """
    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        # Project input embeddings to a lower-dimensional latent space
        self.project_q = nn.Linear(d_model, hidden_dim, bias=True)
        self.project_k = nn.Linear(d_model, hidden_dim, bias=True)
        
        # Activation function for the latent space
        self.act = nn.Tanh()
        
        # Initialize with small random values instead of zeros
        # This ensures gradients can flow from the start
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Use small random initialization (not zero!)
        # Zero initialization prevents gradient flow
        nn.init.normal_(self.project_q.weight, mean=0.0, std=0.01)
        if self.project_q.bias is not None:
            nn.init.zeros_(self.project_q.bias)
        nn.init.normal_(self.project_k.weight, mean=0.0, std=0.01)
        if self.project_k.bias is not None:
            nn.init.zeros_(self.project_k.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
                           In GroupSelfAttention, seq_len corresponds to the number of variates (time series).

        Returns:
            bias: (batch_size, 1, seq_len, seq_len)
                  Additive bias for attention scores.
        """
        # 1. Project to lower dimension latent representation
        # Shape: (batch, seq, hidden_dim)
        q = self.act(self.project_q(hidden_states))
        k = self.act(self.project_k(hidden_states))
        
        # 2. Compute pair-wise importance scores (bilinear interaction)
        # (batch, seq, hidden_dim) @ (batch, hidden_dim, seq) -> (batch, seq, seq)
        relevance_scores = torch.matmul(q, k.transpose(-2, -1))
        
        # 3. Scale to prevent dominating original attention
        # Start with small influence
        relevance_scores = relevance_scores * 0.1
        
        # 4. Reshape for broadcasting over attention heads
        # Target shape: (batch_size, n_heads, seq_len, seq_len)
        return relevance_scores.unsqueeze(1)


class AdaptiveGroupAttention(GroupSelfAttention):
    """
    Extends Chronos GroupSelfAttention with Adaptive Relevance Bias.
    """
    def __init__(self, config: Chronos2CoreConfig):
        super().__init__(config)
        # Initialize the relevance bias module
        self.relevance_bias = RelevanceBias(d_model=config.d_model)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool = False
    ) -> AttentionOutput:
        # flip time and batch axes because attention operates along dim=-2
        hidden_states = rearrange(hidden_states, "batch time d -> time batch d")
        normed_hidden_states = self.layer_norm(hidden_states)

        # --- [NEW] Compute Relevance Bias ---
        bias = self.relevance_bias(normed_hidden_states)
        
        # --- [FIXED] Properly combine bias with existing mask ---
        # The mask should be added to attention logits, not multiplied
        # If no mask exists, create one with zeros
        if attention_mask is not None:
            # Add bias to existing mask
            combined_mask = attention_mask + bias
        else:
            # Use bias as the mask
            combined_mask = bias
            
        # Call original MHA with the modified mask
        attention_output: AttentionOutput = self.self_attention(
            normed_hidden_states, mask=combined_mask, output_attentions=output_attentions
        )
        
        # Residual connection matching original Chronos-2 exactly
        hidden_states = hidden_states + self.dropout(attention_output[0])
        
        # flip time and batch axes back to their original position
        hidden_states = rearrange(hidden_states, "time batch d -> batch time d")

        return AttentionOutput(hidden_states=hidden_states, attn_weights=attention_output.attn_weights)


def replace_chronos_attention(pipeline):
    """
    Replaces GroupSelfAttention layers in a loaded ChronosPipeline 
    with AdaptiveGroupAttention layers.
    
    Args:
        pipeline: The loaded ChronosPipeline or Chronos2Pipeline.
        
    Returns:
        The pipeline with modified attention layers.
    """
    print("Replacing attention layers with AdaptiveGroupAttention...")
    
    # Navigate to the encoder blocks
    # Structure: pipeline -> model (ChronosModel) -> model (Chronos2Model) -> encoder -> block
    try:
        # Access inner model
        if hasattr(pipeline, "model") and hasattr(pipeline.model, "model"):
             inner_model = pipeline.model.model
        elif hasattr(pipeline, "model"):
             # Sometimes it might be one level deep
             inner_model = pipeline.model
        else:
             # Maybe raw model passed
             inner_model = pipeline
             
        if not hasattr(inner_model, "encoder"):
            print("Warning: Could not find encoder in model. Aborting replacement.")
            print(f"Inner model type: {type(inner_model)}")
            return pipeline
            
        blocks = inner_model.encoder.block
        count = 0
        print(f"Found {len(blocks)} encoder blocks.")
        for i, block in enumerate(blocks):
            # layer[1] is GroupSelfAttention in Chronos2EncoderBlock
            if len(block.layer) > 1:
                layer_to_check = block.layer[1]
                
                if isinstance(layer_to_check, GroupSelfAttention):
                    old_layer = layer_to_check
                    
                    # Create new layer using the model's config
                    new_layer = AdaptiveGroupAttention(inner_model.config)
                    
                    # Load existing weights (strict=False to ignore new relevance_bias params)
                    new_layer.load_state_dict(old_layer.state_dict(), strict=False)
                    
                    # Move new layer to the same device/dtype as the old one
                    device = next(old_layer.parameters()).device
                    dtype = next(old_layer.parameters()).dtype
                    new_layer.to(device=device, dtype=dtype)
                    
                    # Replace
                    block.layer[1] = new_layer
                    count += 1
        
        print(f"Successfully replaced {count} GroupSelfAttention layers.")
        
    except Exception as e:
        print(f"Error replacing layers: {e}")
        import traceback
        traceback.print_exc()
        
    return pipeline
