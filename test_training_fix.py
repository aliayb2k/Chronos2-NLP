"""
Quick test to verify the training bug is fixed.
This runs just a few training steps to confirm no errors.
"""

from train_on_ett import train_on_ett

print("Running quick test (50 steps) to verify fix...")
print("="*70)

try:
    pipeline, log = train_on_ett(
        max_steps=50,
        val_every=25,
        batch_size=8,  # Smaller for faster test
        early_stopping_patience=5
    )
    
    print("\n" + "="*70)
    print("✅ TEST PASSED!")
    print("="*70)
    print(f"Successfully completed {len(log['train_loss'])} training steps")
    print(f"Training losses: {log['train_loss'][:5]}...")
    print("\nThe bug is fixed! You can now run full training with:")
    print("  python train_on_ett.py")
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
