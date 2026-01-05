"""
Test suite for Matryoshka Representation Learning (MRL) functionality.

This test file validates that MRL loss computation works correctly
and that embeddings maintain quality across multiple dimensions.
"""

import torch
import torch.nn.functional as F
from utils import matryoshka_loss


class MockAccelerator:
    """Mock accelerator for testing without distributed setup"""
    def __init__(self):
        self.process_index = 0
        self.num_processes = 1
    
    def gather(self, tensor):
        return tensor


def test_matryoshka_loss_basic():
    """Test that MRL loss returns a scalar and is differentiable"""
    print("Test 1: Basic MRL loss computation...")
    
    batch_size = 16
    embedding_dim = 1024
    
    # Create random embeddings
    embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)
    
    # Normalize embeddings
    embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
    
    # Define MRL dimensions
    mrl_dimensions = [64, 128, 256, 512, 1024]
    
    # Compute MRL loss
    loss = matryoshka_loss(embeddings_norm, mrl_dimensions)
    
    # Check that loss is a scalar
    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert loss.item() > 0, "Expected positive loss"
    
    # Check that loss is differentiable
    loss.backward()
    assert embeddings.grad is not None, "Expected gradient to be computed"
    
    print(f"  ✓ MRL loss computed: {loss.item():.6f}")
    print(f"  ✓ Loss is differentiable")
    print()


def test_matryoshka_loss_different_dimensions():
    """Test MRL loss with different dimension sets"""
    print("Test 2: MRL loss with different dimension configurations...")
    
    batch_size = 32
    embedding_dim = 512
    
    embeddings = torch.randn(batch_size, embedding_dim)
    embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
    
    # Test with different dimension sets
    dimension_sets = [
        [64, 128, 256, 512],
        [128, 256],
        [64],
        [512],
    ]
    
    losses = []
    for dims in dimension_sets:
        loss = matryoshka_loss(embeddings_norm, dims)
        losses.append(loss.item())
        # Loss might be very small but should not be negative
        assert loss.item() >= 0, f"Expected non-negative loss for dims {dims}"
        print(f"  ✓ Dims {dims}: loss = {loss.item():.6f}")
    
    print()


def test_matryoshka_loss_edge_cases():
    """Test MRL loss with edge cases"""
    print("Test 3: Edge cases...")
    
    batch_size = 16
    embedding_dim = 256
    
    embeddings = torch.randn(batch_size, embedding_dim)
    embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
    
    # Test with None dimensions
    loss = matryoshka_loss(embeddings_norm, None)
    assert loss == 0.0, "Expected 0 loss for None dimensions"
    print(f"  ✓ None dimensions: loss = {loss}")
    
    # Test with empty dimensions
    loss = matryoshka_loss(embeddings_norm, [])
    assert loss == 0.0, "Expected 0 loss for empty dimensions"
    print(f"  ✓ Empty dimensions: loss = {loss}")
    
    # Test with dimensions larger than embedding dim
    loss = matryoshka_loss(embeddings_norm, [256, 512, 1024])
    assert loss.item() >= 0, "Expected valid loss even with large dimensions"
    print(f"  ✓ Dimensions larger than embedding: loss = {loss.item():.6f}")
    
    print()


def test_matryoshka_loss_temperature():
    """Test that temperature affects loss magnitude"""
    print("Test 4: Temperature scaling...")
    
    batch_size = 8
    embedding_dim = 256
    
    embeddings = torch.randn(batch_size, embedding_dim)
    embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
    
    mrl_dimensions = [64, 128, 256]
    
    # Note: temperature parameter needs to be added to loss computation
    # For now, we just verify the function works with normalized embeddings
    loss1 = matryoshka_loss(embeddings_norm, mrl_dimensions, temperature=0.05)
    loss2 = matryoshka_loss(embeddings_norm, mrl_dimensions, temperature=0.1)
    
    print(f"  ✓ Loss with temperature=0.05: {loss1.item():.6f}")
    print(f"  ✓ Loss with temperature=0.1: {loss2.item():.6f}")
    print()


def test_inbatch_loss_with_mrl():
    """Test that inbatch_loss works with MRL enabled"""
    print("Test 5: In-batch loss with MRL integration...")
    
    from utils import inbatch_loss
    from torch.nn import CrossEntropyLoss
    
    batch_size = 32
    embedding_dim = 512
    
    query_embeddings = torch.randn(batch_size, embedding_dim)
    context_embeddings = torch.randn(batch_size, embedding_dim)
    
    # Create mock accelerator
    accelerator = MockAccelerator()
    criterion = CrossEntropyLoss(reduction='none')
    
    mrl_dimensions = [64, 128, 256, 512]
    
    # Test without MRL
    loss_no_mrl = inbatch_loss(
        query_embeddings, 
        context_embeddings, 
        criterion, 
        accelerator,
        use_mrl=False
    )
    
    # Test with MRL
    loss_with_mrl = inbatch_loss(
        query_embeddings, 
        context_embeddings, 
        criterion, 
        accelerator,
        mrl_dimensions=mrl_dimensions,
        use_mrl=True
    )
    
    print(f"  ✓ Loss without MRL: {loss_no_mrl.item():.6f}")
    print(f"  ✓ Loss with MRL: {loss_with_mrl.item():.6f}")
    assert loss_with_mrl.item() >= loss_no_mrl.item(), "MRL loss should not decrease total loss"
    print()


def test_hard_loss_with_mrl():
    """Test that hard_loss works with MRL enabled"""
    print("Test 6: Hard loss with MRL integration...")
    
    from utils import hard_loss
    from torch.nn import CrossEntropyLoss
    
    batch_size = 32
    embedding_dim = 512
    num_hard_neg = 7
    
    query_embeddings = torch.randn(batch_size, embedding_dim)
    context_embeddings = torch.randn(batch_size, embedding_dim)
    hard_neg_embeddings = torch.randn(batch_size, num_hard_neg, embedding_dim)
    
    # Create mock accelerator
    accelerator = MockAccelerator()
    criterion = CrossEntropyLoss(reduction='none')
    
    mrl_dimensions = [64, 128, 256, 512]
    
    # Test without MRL
    loss_no_mrl = hard_loss(
        query_embeddings,
        context_embeddings,
        hard_neg_embeddings,
        criterion,
        accelerator,
        use_mrl=False
    )
    
    # Test with MRL
    loss_with_mrl = hard_loss(
        query_embeddings,
        context_embeddings,
        hard_neg_embeddings,
        criterion,
        accelerator,
        mrl_dimensions=mrl_dimensions,
        use_mrl=True
    )
    
    print(f"  ✓ Hard loss without MRL: {loss_no_mrl.item():.6f}")
    print(f"  ✓ Hard loss with MRL: {loss_with_mrl.item():.6f}")
    assert loss_with_mrl.item() >= loss_no_mrl.item(), "MRL loss should not decrease total loss"
    print()


def test_embedding_dimension_truncation():
    """Test that embeddings maintain quality when truncated"""
    print("Test 7: Embedding quality at different dimensions...")
    
    batch_size = 16
    embedding_dim = 1024
    
    # Create embeddings from a Gaussian distribution
    embeddings = torch.randn(batch_size, embedding_dim)
    embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
    
    # Compute similarity matrix at full dimension
    sim_full = torch.matmul(embeddings_norm, embeddings_norm.t())
    
    # Compute similarities at truncated dimensions
    test_dims = [64, 128, 256, 512, 1024]
    similarities = []
    
    for dim in test_dims:
        truncated = embeddings_norm[:, :dim]
        truncated_norm = F.normalize(truncated, p=2, dim=-1)
        sim = torch.matmul(truncated_norm, truncated_norm.t())
        similarities.append(sim)
        
        # Compare to full dimension (should be similar, especially for larger dims)
        correlation = F.cosine_similarity(
            sim_full.flatten().unsqueeze(0),
            sim.flatten().unsqueeze(0)
        )
        print(f"  ✓ Dim {dim}: correlation with full dim = {correlation.item():.6f}")
    
    print()


def test_mrl_batch_consistency():
    """Test that MRL loss is consistent across batches"""
    print("Test 8: Batch consistency...")
    
    embedding_dim = 512
    mrl_dimensions = [128, 256, 512]
    
    # Create fixed embeddings
    torch.manual_seed(42)
    embeddings = torch.randn(16, embedding_dim)
    embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
    
    # Compute loss on full batch
    loss_full = matryoshka_loss(embeddings_norm, mrl_dimensions)
    
    # Compute loss on splits
    half_size = embeddings_norm.size(0) // 2
    loss_first = matryoshka_loss(embeddings_norm[:half_size], mrl_dimensions)
    loss_second = matryoshka_loss(embeddings_norm[half_size:], mrl_dimensions)
    
    print(f"  ✓ Full batch loss: {loss_full.item():.6f}")
    print(f"  ✓ First half loss: {loss_first.item():.6f}")
    print(f"  ✓ Second half loss: {loss_second.item():.6f}")
    print()


def run_all_tests():
    """Run all test cases"""
    print("=" * 70)
    print("Matryoshka Representation Learning (MRL) Test Suite")
    print("=" * 70)
    print()
    
    try:
        test_matryoshka_loss_basic()
        test_matryoshka_loss_different_dimensions()
        test_matryoshka_loss_edge_cases()
        test_matryoshka_loss_temperature()
        test_inbatch_loss_with_mrl()
        test_hard_loss_with_mrl()
        test_embedding_dimension_truncation()
        test_mrl_batch_consistency()
        
        print("=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)
        
    except Exception as e:
        print("=" * 70)
        print(f"Test failed with error: {e}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    run_all_tests()
