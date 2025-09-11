#!/usr/bin/env python
"""
Simple runner for HWO-SLAPS pipeline with Module 4 detection support.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from hwoslaps.pipeline import run_enhanced_pipeline


def main():
    parser = argparse.ArgumentParser(description='Run HWO-SLAPS pipeline')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    args = parser.parse_args()
    
    # Run enhanced pipeline (automatically detects standard vs detection mode)
    run_enhanced_pipeline(args.config, verbose=not args.quiet)


if __name__ == '__main__':
    main()