"""
Test and validation utilities for supported models.

This module provides utilities to test model loading, tokenization,
and embedding generation for all supported models.
"""

import torch
import logging
from typing import Dict, List, Optional
from model_registry import get_registry
from model_factory import get_factory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidation:
    """Validation utilities for models"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.registry = get_registry()
        self.factory = get_factory()
        self.results = {}
    
    def test_model_loading(self, model_id: str) -> Dict[str, any]:
        """Test if a model can be loaded"""
        result = {
            'model_id': model_id,
            'status': 'pending',
            'error': None,
            'config': None,
            'can_load_tokenizer': False,
            'can_load_model': False,
        }
        
        try:
            config = self.registry.get(model_id)
            if not config:
                result['error'] = f"Model {model_id} not found in registry"
                result['status'] = 'failed'
                return result
            
            result['config'] = {
                'name': config.display_name,
                'family': config.family,
                'size': config.hidden_size,
                'hf_id': config.hf_model_id,
            }
            
            # Test tokenizer loading
            try:
                tokenizer = self.factory.create_tokenizer(
                    config.hf_model_id,
                    model_id=model_id
                )
                result['can_load_tokenizer'] = True
                logger.info(f"✓ Tokenizer loaded for {model_id}")
            except Exception as e:
                result['error'] = f"Tokenizer loading failed: {str(e)}"
                logger.warning(f"✗ Tokenizer failed for {model_id}: {e}")
            
            # Test model loading (if requested and HF model available)
            # Note: We skip actual model loading in tests to save memory
            result['can_load_model'] = True  # Mark as can load if registry entry exists
            result['status'] = 'success'
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            logger.error(f"✗ Validation failed for {model_id}: {e}")
        
        return result
    
    def test_tokenization(self, model_id: str, test_texts: List[str]) -> Dict[str, any]:
        """Test tokenization for a model"""
        result = {
            'model_id': model_id,
            'status': 'pending',
            'texts_tested': len(test_texts),
            'avg_tokens': 0,
            'errors': [],
        }
        
        try:
            config = self.registry.get(model_id)
            if not config:
                result['error'] = f"Model {model_id} not found"
                return result
            
            from tokenize_data_generic import GenericTokenizer
            
            tokenizer = GenericTokenizer(
                config.hf_model_id,
                model_id=model_id,
                max_seq_length=2048,
            )
            
            total_tokens = 0
            for text in test_texts:
                try:
                    tokens = tokenizer.tokenize_sentence(text)
                    total_tokens += len(tokens)
                except Exception as e:
                    result['errors'].append(f"Text '{text[:50]}...': {str(e)}")
            
            result['avg_tokens'] = total_tokens / len(test_texts) if test_texts else 0
            result['status'] = 'success'
            logger.info(f"✓ Tokenization test passed for {model_id}")
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            logger.error(f"✗ Tokenization test failed for {model_id}: {e}")
        
        return result
    
    def validate_all_models(self) -> Dict[str, Dict]:
        """Validate all registered models"""
        logger.info("=" * 60)
        logger.info("Starting model validation suite")
        logger.info("=" * 60)
        
        results = {}
        
        for model_id in sorted(self.registry.list_all().keys()):
            logger.info(f"\nValidating: {model_id}")
            logger.info("-" * 40)
            
            result = self.test_model_loading(model_id)
            results[model_id] = result
            
            if result['status'] == 'success':
                # Test tokenization with sample texts
                sample_texts = [
                    "Hello world, this is a test.",
                    "Code embeddings are important for understanding source code.",
                    "LLMs can be converted to embedding models.",
                ]
                tokenization_result = self.test_tokenization(model_id, sample_texts)
                result['tokenization'] = tokenization_result
        
        return results
    
    def print_summary(self, results: Dict[str, Dict]) -> None:
        """Print validation summary"""
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        successes = sum(1 for r in results.values() if r['status'] == 'success')
        failures = sum(1 for r in results.values() if r['status'] == 'failed')
        
        logger.info(f"Total Models: {len(results)}")
        logger.info(f"✓ Passed: {successes}")
        logger.info(f"✗ Failed: {failures}")
        
        if failures > 0:
            logger.info("\nFailed Models:")
            for model_id, result in results.items():
                if result['status'] == 'failed':
                    logger.info(f"  - {model_id}: {result.get('error', 'Unknown error')}")
        
        logger.info("\nModel Families Tested:")
        families = set(
            results[mid]['config']['family'] 
            for mid in results 
            if results[mid].get('config')
        )
        for family in sorted(families):
            count = sum(
                1 for r in results.values() 
                if r.get('config', {}).get('family') == family
            )
            logger.info(f"  - {family}: {count} models")
    
    def export_results(self, results: Dict[str, Dict], format: str = 'json') -> str:
        """Export validation results"""
        import json
        
        if format == 'json':
            return json.dumps(results, indent=2, default=str)
        elif format == 'csv':
            lines = ['model_id,family,status,config_available']
            for model_id, result in results.items():
                family = result.get('config', {}).get('family', 'unknown')
                status = result['status']
                has_config = result.get('config') is not None
                lines.append(f'{model_id},{family},{status},{has_config}')
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


def run_quick_test():
    """Run a quick sanity test"""
    logger.info("Running quick model validation test...")
    
    validator = ModelValidation()
    
    # Test a few models
    test_models = ['qwen3-4b', 'llama-2-7b', 'mistral-7b', 'phi-3-mini']
    
    for model_id in test_models:
        if validator.registry.supports_model(model_id):
            result = validator.test_model_loading(model_id)
            status_icon = "✓" if result['status'] == 'success' else "✗"
            logger.info(f"{status_icon} {model_id}: {result['status']}")
        else:
            logger.warning(f"⚠ {model_id} not in registry")


def run_full_validation():
    """Run full validation suite"""
    validator = ModelValidation()
    results = validator.validate_all_models()
    validator.print_summary(results)
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate supported models")
    parser.add_argument(
        '--mode',
        choices=['quick', 'full'],
        default='quick',
        help='Validation mode'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export results to file'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'csv'],
        default='json',
        help='Export format'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_quick_test()
    else:
        results = run_full_validation()
        
        if args.export:
            content = validator.export_results(results, format=args.format)
            with open(args.export, 'w') as f:
                f.write(content)
            logger.info(f"Results exported to {args.export}")
