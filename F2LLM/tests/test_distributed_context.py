"""
Unit tests for DistributedContext

Tests the abstraction layer for both Accelerate and Ray Train backends.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from utils import DistributedContext


class TestDistributedContextAccelerate(unittest.TestCase):
    """Test DistributedContext with Accelerate backend"""

    @patch('utils.Accelerator')
    def test_init_accelerate(self, MockAccelerator):
        """Test Accelerate backend initialization"""
        mock_acc = MockAccelerator.return_value
        mock_acc.process_index = 0
        mock_acc.num_processes = 4
        mock_acc.local_process_index = 0

        ctx = DistributedContext(backend='accelerate')

        self.assertEqual(ctx.backend, 'accelerate')
        self.assertEqual(ctx.rank, 0)
        self.assertEqual(ctx.world_size, 4)
        self.assertEqual(ctx.local_rank, 0)
        MockAccelerator.assert_called_once()

    @patch('utils.Accelerator')
    def test_is_main_process(self, MockAccelerator):
        """Test main process detection"""
        mock_acc = MockAccelerator.return_value
        mock_acc.process_index = 0
        mock_acc.num_processes = 4
        mock_acc.local_process_index = 0

        ctx = DistributedContext(backend='accelerate')
        self.assertTrue(ctx.is_main_process())

        # Non-main process
        mock_acc.process_index = 1
        ctx2 = DistributedContext(backend='accelerate')
        self.assertFalse(ctx2.is_main_process())

    @patch('utils.Accelerator')
    def test_gather_accelerate(self, MockAccelerator):
        """Test gather operation with Accelerate"""
        mock_acc = MockAccelerator.return_value
        mock_acc.process_index = 0
        mock_acc.num_processes = 1

        tensor = torch.randn(4, 128)
        mock_acc.gather.return_value = tensor

        ctx = DistributedContext(backend='accelerate')
        result = ctx.gather(tensor)

        mock_acc.gather.assert_called_once_with(tensor)
        torch.testing.assert_close(result, tensor)

    @patch('utils.Accelerator')
    def test_wait_for_everyone(self, MockAccelerator):
        """Test synchronization barrier"""
        mock_acc = MockAccelerator.return_value
        mock_acc.process_index = 0
        mock_acc.num_processes = 4

        ctx = DistributedContext(backend='accelerate')
        ctx.wait_for_everyone()

        mock_acc.wait_for_everyone.assert_called_once()


class TestDistributedContextRay(unittest.TestCase):
    """Test DistributedContext with Ray backend"""

    @patch('ray.train.get_context')
    def test_init_ray(self, mock_get_context):
        """Test Ray backend initialization"""
        mock_ctx = Mock()
        mock_ctx.get_world_rank.return_value = 1
        mock_ctx.get_world_size.return_value = 8
        mock_ctx.get_local_rank.return_value = 1
        mock_get_context.return_value = mock_ctx

        ctx = DistributedContext(backend='ray')

        self.assertEqual(ctx.backend, 'ray')
        self.assertEqual(ctx.rank, 1)
        self.assertEqual(ctx.world_size, 8)
        self.assertEqual(ctx.local_rank, 1)

    @patch('ray.train.get_context')
    def test_is_main_process_ray(self, mock_get_context):
        """Test main process detection with Ray"""
        mock_ctx = Mock()
        mock_ctx.get_world_rank.return_value = 0
        mock_ctx.get_world_size.return_value = 8
        mock_ctx.get_local_rank.return_value = 0
        mock_get_context.return_value = mock_ctx

        ctx = DistributedContext(backend='ray')
        self.assertTrue(ctx.is_main_process())

        # Non-main process
        mock_ctx.get_world_rank.return_value = 1
        ctx2 = DistributedContext(backend='ray')
        self.assertFalse(ctx2.is_main_process())

    @patch('torch.distributed.is_initialized')
    @patch('torch.distributed.all_gather')
    @patch('ray.train.get_context')
    def test_gather_ray(self, mock_get_context, mock_all_gather, mock_is_init):
        """Test gather operation with Ray"""
        mock_ctx = Mock()
        mock_ctx.get_world_rank.return_value = 0
        mock_ctx.get_world_size.return_value = 2
        mock_ctx.get_local_rank.return_value = 0
        mock_get_context.return_value = mock_ctx
        mock_is_init.return_value = True

        tensor = torch.randn(4, 128)

        # Mock all_gather behavior
        def all_gather_side_effect(tensor_list, tensor):
            for i, t in enumerate(tensor_list):
                t.copy_(tensor)
        mock_all_gather.side_effect = all_gather_side_effect

        ctx = DistributedContext(backend='ray')
        result = ctx.gather(tensor)

        # Should concatenate tensors
        self.assertEqual(result.shape[0], tensor.shape[0] * 2)
        mock_all_gather.assert_called_once()

    @patch('torch.distributed.is_initialized')
    @patch('ray.train.get_context')
    def test_gather_ray_not_initialized(self, mock_get_context, mock_is_init):
        """Test gather when distributed not initialized"""
        mock_ctx = Mock()
        mock_ctx.get_world_rank.return_value = 0
        mock_ctx.get_world_size.return_value = 1
        mock_ctx.get_local_rank.return_value = 0
        mock_get_context.return_value = mock_ctx
        mock_is_init.return_value = False

        tensor = torch.randn(4, 128)

        ctx = DistributedContext(backend='ray')
        result = ctx.gather(tensor)

        # Should return original tensor when not initialized
        torch.testing.assert_close(result, tensor)


class TestDistributedContextAutoDetect(unittest.TestCase):
    """Test auto-detection of backend"""

    @patch('ray.train.get_context')
    @patch('utils.Accelerator')
    def test_auto_detect_ray(self, MockAccelerator, mock_get_context):
        """Test auto-detection chooses Ray when available"""
        mock_ctx = Mock()
        mock_ctx.get_world_rank.return_value = 0
        mock_ctx.get_world_size.return_value = 4
        mock_ctx.get_local_rank.return_value = 0
        mock_get_context.return_value = mock_ctx

        ctx = DistributedContext(backend='auto')

        self.assertEqual(ctx.backend, 'ray')
        MockAccelerator.assert_not_called()

    @patch('ray.train.get_context', side_effect=ImportError)
    @patch('utils.Accelerator')
    def test_auto_detect_accelerate(self, MockAccelerator, mock_get_context):
        """Test auto-detection falls back to Accelerate"""
        mock_acc = MockAccelerator.return_value
        mock_acc.process_index = 0
        mock_acc.num_processes = 4
        mock_acc.local_process_index = 0

        ctx = DistributedContext(backend='auto')

        self.assertEqual(ctx.backend, 'accelerate')
        MockAccelerator.assert_called_once()


class TestDistributedContextHelpers(unittest.TestCase):
    """Test helper methods"""

    @patch('utils.Accelerator')
    def test_prepare_accelerate(self, MockAccelerator):
        """Test prepare method with Accelerate"""
        mock_acc = MockAccelerator.return_value
        mock_acc.process_index = 0
        mock_acc.num_processes = 1

        model = Mock()
        optimizer = Mock()
        mock_acc.prepare.return_value = (model, optimizer)

        ctx = DistributedContext(backend='accelerate')
        result = ctx.prepare(model, optimizer)

        mock_acc.prepare.assert_called_once_with(model, optimizer)
        self.assertEqual(len(result), 2)

    @patch('ray.train.get_context')
    def test_prepare_ray(self, mock_get_context):
        """Test prepare method with Ray (no-op)"""
        mock_ctx = Mock()
        mock_ctx.get_world_rank.return_value = 0
        mock_ctx.get_world_size.return_value = 1
        mock_ctx.get_local_rank.return_value = 0
        mock_get_context.return_value = mock_ctx

        model = Mock()
        optimizer = Mock()

        ctx = DistributedContext(backend='ray')
        result = ctx.prepare(model, optimizer)

        # Ray doesn't transform objects
        self.assertEqual(result, (model, optimizer))

    @patch('utils.Accelerator')
    def test_unwrap_model_accelerate(self, MockAccelerator):
        """Test unwrap_model with Accelerate"""
        mock_acc = MockAccelerator.return_value
        mock_acc.process_index = 0
        mock_acc.num_processes = 1

        model = Mock()
        unwrapped = Mock()
        mock_acc.unwrap_model.return_value = unwrapped

        ctx = DistributedContext(backend='accelerate')
        result = ctx.unwrap_model(model)

        mock_acc.unwrap_model.assert_called_once_with(model)
        self.assertEqual(result, unwrapped)

    @patch('ray.train.get_context')
    def test_unwrap_model_ray_with_module(self, mock_get_context):
        """Test unwrap_model with Ray (DDP wrapped)"""
        mock_ctx = Mock()
        mock_ctx.get_world_rank.return_value = 0
        mock_ctx.get_world_size.return_value = 1
        mock_ctx.get_local_rank.return_value = 0
        mock_get_context.return_value = mock_ctx

        # Simulate DDP-wrapped model
        inner_model = Mock()
        wrapped_model = Mock()
        wrapped_model.module = inner_model

        ctx = DistributedContext(backend='ray')
        result = ctx.unwrap_model(wrapped_model)

        self.assertEqual(result, inner_model)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
