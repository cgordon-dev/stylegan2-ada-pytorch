#!/usr/bin/env python3
"""Unit tests for StyleGAN2-ADA workload modules."""

import os
import sys
import unittest
import tempfile
import json
import yaml
from unittest.mock import patch, MagicMock, mock_open

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import modules to test
from workload_modules.training_workload import load_dataset, load_config_file, setup_training_config
from workload_modules.inference_workload import num_range, process_industry_dataset
from workload_modules.fine_tuning_optimization import setup_fine_tuning_config
from workload_modules.latent_vector_optimization import get_perceptual_loss
from workload_modules.mixed_precision_optimization import setup_mixed_precision_config


class TestWorkloadModules(unittest.TestCase):
    """Test cases for StyleGAN2-ADA workload modules."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        
        # Sample config for testing
        self.sample_config = {
            'dataset': {
                'path': './datasets/test',
                'image_size': 256,
                'use_labels': True,
                'mirror': True
            },
            'gpus': 1,
            'snap': 50,
            'metrics': ['fid50k_full'],
            'cfg': 'auto',
            'kimg': 10000,
            'aug': 'ada',
            'resume': 'ffhq256',
            'mixed_precision_mode': 'default'
        }
        
        # Create a sample config file
        self.config_file = os.path.join(self.test_dir.name, 'test_config.yaml')
        with open(self.config_file, 'w') as f:
            yaml.dump(self.sample_config, f)

    def tearDown(self):
        """Tear down test fixtures."""
        self.test_dir.cleanup()

    def test_load_dataset(self):
        """Test loading a dataset configuration."""
        dataset_kwargs = load_dataset(
            dataset_path='./datasets/test',
            image_size=256,
            use_labels=True,
            max_size=None,
            xflip=True
        )
        
        self.assertEqual(dataset_kwargs.path, './datasets/test')
        self.assertEqual(dataset_kwargs.resolution, 256)
        self.assertEqual(dataset_kwargs.use_labels, True)
        self.assertEqual(dataset_kwargs.xflip, True)

    def test_load_config_file(self):
        """Test loading configuration from a file."""
        config = load_config_file(self.config_file)
        
        self.assertEqual(config['dataset']['path'], './datasets/test')
        self.assertEqual(config['dataset']['image_size'], 256)
        self.assertEqual(config['gpus'], 1)

    def test_num_range(self):
        """Test number range parsing."""
        # Test comma-separated list
        result = num_range('1,3,5')
        self.assertEqual(result, [1, 3, 5])
        
        # Test range with hyphen
        result = num_range('1-5')
        self.assertEqual(result, [1, 2, 3, 4, 5])

    @patch('dnnlib.EasyDict')
    def test_setup_training_config(self, mock_easydict):
        """Test setting up training configuration."""
        mock_easydict.return_value = MagicMock()
        
        args = setup_training_config(self.sample_config)
        
        # Verify the EasyDict was created
        mock_easydict.assert_called()

    @patch('dnnlib.EasyDict')
    def test_setup_fine_tuning_config(self, mock_easydict):
        """Test setting up fine-tuning configuration."""
        mock_easydict.return_value = MagicMock()
        
        # Update config for fine-tuning
        config = self.sample_config.copy()
        config['freezed'] = 4
        
        args = setup_fine_tuning_config(config)
        
        # Verify the EasyDict was created
        mock_easydict.assert_called()

    @patch('dnnlib.EasyDict')
    def test_setup_mixed_precision_config(self, mock_easydict):
        """Test setting up mixed precision configuration."""
        mock_easydict.return_value = MagicMock()
        
        args = setup_mixed_precision_config(self.sample_config)
        
        # Verify the EasyDict was created
        mock_easydict.assert_called()

    def test_get_perceptual_loss(self):
        """Test getting perceptual loss function."""
        # Test L2 loss
        l2_loss = get_perceptual_loss('none')
        self.assertTrue(callable(l2_loss))
        
        # Test LPIPS loss (this will be mocked)
        with patch('lpips.LPIPS'):
            lpips_loss = get_perceptual_loss('lpips')
            self.assertTrue(callable(lpips_loss))

    def test_process_industry_dataset(self):
        """Test processing industry dataset - just test function existence."""
        # This is a more complex function that would require more extensive mocking
        # For this test, we just confirm the function exists
        self.assertTrue(callable(process_industry_dataset))


if __name__ == '__main__':
    unittest.main()