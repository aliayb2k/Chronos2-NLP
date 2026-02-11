"""
Test script to verify adaptive attention weights can be trained.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from adaptive_attention import RelevanceBias

# Test 1: Verify non-zero initialization
print("="*60)
print("TEST 1: Checking Initialization")
print("="*60)

model = RelevanceBias(d_model=32, hidden_dim=16)

# Check initial weights
q_weight_sum = model.project_q.weight.abs().sum().item()
k_weight_sum = model.project_k.weight.abs().sum().item()

print(f"Query projection weight sum: {q_weight_sum:.6f}")
print(f"Key projection weight sum: {k_weight_sum:.6f}")

if q_weight_sum > 0 and k_weight_sum > 0:
    print("✅ PASS: Weights are non-zero!")
else:
    print("❌ FAIL: Weights are still zero!")
    exit(1)

# Test 2: Verify gradients flow
print("\n" + "="*60)
print("TEST 2: Checking Gradient Flow")
print("="*60)

# Create dummy input
x = torch.randn(2, 8, 32)  # (batch=2, seq=8, d_model=32)
target = torch.randn(2, 1, 8, 8)  # Dummy target bias

# Forward pass
output = model(x)
print(f"Output shape: {output.shape}")

# Compute loss
loss = nn.MSELoss()(output, target)
print(f"Loss: {loss.item():.6f}")

# Backward pass
loss.backward()

# Check gradients
q_grad = model.project_q.weight.grad
k_grad = model.project_k.weight.grad

if q_grad is not None and k_grad is not None:
    q_grad_norm = q_grad.norm().item()
    k_grad_norm = k_grad.norm().item()
    print(f"Query gradient norm: {q_grad_norm:.6f}")
    print(f"Key gradient norm: {k_grad_norm:.6f}")
    
    if q_grad_norm > 0 and k_grad_norm > 0:
        print("✅ PASS: Gradients are flowing!")
    else:
        print("❌ FAIL: Gradients are zero!")
        exit(1)
else:
    print("❌ FAIL: No gradients computed!")
    exit(1)

# Test 3: Verify weights can be updated
print("\n" + "="*60)
print("TEST 3: Checking Weight Updates")
print("="*60)

# Store initial weights
initial_q = model.project_q.weight.clone()
initial_k = model.project_k.weight.clone()

# Create optimizer and train for a few steps
optimizer = optim.Adam(model.parameters(), lr=0.01)

for step in range(10):
    x = torch.randn(2, 8, 32)
    target = torch.randn(2, 1, 8, 8)
    
    optimizer.zero_grad()
    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    optimizer.step()

# Check if weights changed
q_diff = (model.project_q.weight - initial_q).abs().sum().item()
k_diff = (model.project_k.weight - initial_k).abs().sum().item()

print(f"Query weight change: {q_diff:.6f}")
print(f"Key weight change: {k_diff:.6f}")

if q_diff > 0 and k_diff > 0:
    print("✅ PASS: Weights are updating!")
else:
    print("❌ FAIL: Weights are not changing!")
    exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED! ✅")
print("="*60)
print("\nThe adaptive attention module is now trainable!")
print("The issue was zero initialization preventing gradient flow.")
