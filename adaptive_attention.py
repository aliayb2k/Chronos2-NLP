
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
        self.project_q = nn.Linear(d_model, hidden_dim)
        self.project_k = nn.Linear(d_model, hidden_dim)
        
        # Activation function for the latent space
        self.act = nn.Tanh()
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize with zeros to start exactly at "identity" (no bias)
        # This ensures Untrained == Baseline initially.
        nn.init.zeros_(self.project_q.weight)
        if self.project_q.bias is not None:
            nn.init.zeros_(self.project_q.bias)
        nn.init.zeros_(self.project_k.weight)
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
        
        # 3. Reshape for broadcasting over attention heads
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
        
        # Scaling to prevent bias from dominating original attention scores
        # Logits in Chronos-2 are unscaled, so they can be large.
        # We want the bias to be a nudge initially.
        bias = bias * 0.1 

        # --- [NEW] Inject Bias into Mask ---
        if attention_mask is not None:
            combined_mask = attention_mask + bias
        else:
            combined_mask = bias
            
        # Call original MHA with the modified mask
        # Using [0] for hidden_states as in original source
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
            print(f"Inner model match string: {inner_model}")
            return pipeline
            
        blocks = inner_model.encoder.block
        count = 0
        print(f"Found {len(blocks)} encoder blocks.")
        for i, block in enumerate(blocks):
            # layer[1] is GroupSelfAttention in Chronos2EncoderBlock
            # Logic: Check if layer[1] is GroupSelfAttention
            if len(block.layer) > 1:
                layer_to_check = block.layer[1]
                # print(f"Block {i} Layer 1 type: {type(layer_to_check)}")
                
                if isinstance(layer_to_check, GroupSelfAttention):
                    old_layer = layer_to_check
                    
                    # Create new layer using the model's config
                    new_layer = AdaptiveGroupAttention(inner_model.config)
                    
                    # Load existing weights (strict=False to ignore new relevance_bias params)
                    # We preserve the pre-trained self-attention and layer norm weights.
                    new_layer.load_state_dict(old_layer.state_dict(), strict=False)
                    
                    # Move new layer to the same device/dtype as the old one
                    device = next(old_layer.parameters()).device
                    dtype = next(old_layer.parameters()).dtype
                    new_layer.to(device=device, dtype=dtype)
                    
                    # Replace
                    block.layer[1] = new_layer
                    count += 1
                else:
                    print(f"Skipping Block {i} Layer 1: Not GroupSelfAttention (Got {type(layer_to_check).__name__})")
        
        print(f"Successfully replaced {count} GroupSelfAttention layers.")
        
    except Exception as e:
        print(f"Error replacing layers: {e}")
        import traceback
        traceback.print_exc()
        
    return pipeline


if __name__ == "__main__":
    # Test Block
    try:
        print("Testing RelevanceBias...")
        batch_size, seq_len, d_model = 2, 8, 32
        model = RelevanceBias(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        bias = model(x)
        print(f"RelevanceBias Output shape: {bias.shape}")
        
        assert bias.shape == (batch_size, 1, seq_len, seq_len)
        print("RelevanceBias PASS")
        
        print("\nTesting AdaptiveGroupAttention...")
        # Create a dummy config
        class MockConfig:
             def __init__(self, **kwargs):
                 for k, v in kwargs.items():
                     setattr(self, k, v)
        
        # Minimal config matching Chronos requirements
        config = MockConfig(
            d_model=32, num_heads=4, d_kv=8, dropout_rate=0.1, 
            layer_norm_epsilon=1e-6, rope_theta=10000, _attn_implementation='eager',
            dense_act_fn='gelu', d_ff=128, is_gated_act=False
        )
        
        adaptive_attn = AdaptiveGroupAttention(config)
        
        # Input: (Batch=4 variates, Time=10 steps, D=32)
        # Remember GroupAttention operates on Batch dim as sequence
        inputs = torch.randn(4, 10, 32)
        
        # Mask needs to match what MHA expects after swap: (Time, Heads, Batch, Batch) = (10, 4, 4, 4)
        # Usually mask is provided by caller. Let's provide a zero mask.
        mask = torch.zeros(10, 1, 4, 4) 
        
        output = adaptive_attn(inputs, attention_mask=mask)
        
        print(f"AdaptiveGroupAttention Output shape: {output.hidden_states.shape}")
        assert output.hidden_states.shape == inputs.shape
        print("AdaptiveGroupAttention PASS")

        print("\nTesting replace_chronos_attention...")
        # Mock Pipeline Structure
        class MockBlock:
            def __init__(self):
                self.layer = [None, GroupSelfAttention(config)]
        
        class MockEncoder:
            def __init__(self):
                self.block = [MockBlock() for _ in range(2)]
        
        class MockModel:
            def __init__(self):
                self.config = config
                self.encoder = MockEncoder()
                
        class MockPipeline:
            def __init__(self):
                # Structure: pipeline.model.model
                self.model = SimpleNamespace(model=MockModel())
        
        from types import SimpleNamespace
        dummy_pipeline = MockPipeline()
        
        # Run replacement
        replace_chronos_attention(dummy_pipeline)
        
        # Verify
        new_layer = dummy_pipeline.model.model.encoder.block[0].layer[1]
        print(f"New layer type: {type(new_layer)}")
        assert isinstance(new_layer, AdaptiveGroupAttention)
        print("Replacement Logic PASS")

        print("\nSuccess: All modules working correctly!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
