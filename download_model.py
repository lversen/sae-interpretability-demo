#!/usr/bin/env python
"""
GPT Neo Model Downloader (Compatible with older huggingface_hub versions)

This script downloads a GPT Neo model using Hugging Face's snapshot_download function,
with compatibility for various versions of the huggingface_hub library.

Usage:
    python download_model.py --model EleutherAI/gpt-neo-125m --output models/gpt-neo-125m
"""

import os
import sys
import argparse
import time
from pathlib import Path
import importlib
import pkg_resources

# Check huggingface_hub version
try:
    hf_version = pkg_resources.get_distribution("huggingface_hub").version
    print(f"Detected huggingface_hub version: {hf_version}")
except Exception:
    hf_version = "0.0.0"
    print("Could not detect huggingface_hub version, assuming compatibility mode")

# Import required functions with version-aware error handling
try:
    from huggingface_hub import snapshot_download, HfFolder
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install the required packages with:")
    print("pip install transformers huggingface_hub")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Download GPT Neo model with extended timeout")
    parser.add_argument("--model", type=str, default="EleutherAI/gpt-neo-125m",
                       help="Model ID on Hugging Face (default: EleutherAI/gpt-neo-125m)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: ./models/[model_name])")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Request timeout in seconds (default: 300)")
    parser.add_argument("--download-tokenizer-only", action="store_true",
                       help="Download only tokenizer files (much smaller)")
    parser.add_argument("--revision", type=str, default="main",
                       help="Model revision to download (default: main)")
    parser.add_argument("--skip-weights", action="store_true",
                       help="Skip large model weight files (useful to just get config)")
    parser.add_argument("--low-memory", action="store_true",
                       help="Use methods optimized for low memory environments")
    parser.add_argument("--smaller-model", action="store_true",
                       help="Download a smaller model (distilgpt2) instead")
    return parser.parse_args()

def download_model_with_snapshot(
    model_id, 
    output_dir=None, 
    timeout=300,
    download_tokenizer_only=False,
    revision="main",
    skip_weights=False,
    low_memory=False
):
    """
    Download a model using snapshot_download with version compatibility
    """
    if output_dir is None:
        model_name = model_id.split("/")[-1]
        output_dir = os.path.join("models", model_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup patterns to ignore
    ignore_patterns = []
    if skip_weights:
        ignore_patterns.extend(["*.bin", "*.safetensors", "*.pt", "*.h5"])
    
    # Set environment variables for timeouts
    original_timeout = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", None)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout)
    os.environ["TRANSFORMERS_REQUEST_TIMEOUT"] = str(timeout)
    
    try:
        print(f"Downloading {model_id} to {output_dir} (timeout: {timeout}s)")
        t_start = time.time()
        
        # Version-aware parameter handling for snapshot_download
        try:
            # Try with parameters for newer versions first
            snapshot_params = {
                "repo_id": model_id,
                "local_dir": output_dir,
                "local_dir_use_symlinks": False,  # For Windows compatibility
                "revision": revision,
                "ignore_patterns": ignore_patterns,
            }
            
            # Add version-specific parameters
            if pkg_resources.parse_version(hf_version) >= pkg_resources.parse_version("0.5.0"):
                snapshot_params["max_workers"] = 1
            
            if pkg_resources.parse_version(hf_version) >= pkg_resources.parse_version("0.8.0"):
                snapshot_params["request_timeout"] = timeout
            
            # Call snapshot_download with appropriate parameters
            print("Downloading model configuration and files...")
            model_path = snapshot_download(**snapshot_params)
            
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                # Fallback for older versions
                print(f"Using compatibility mode due to: {e}")
                model_path = snapshot_download(
                    repo_id=model_id,
                    local_dir=output_dir,
                    revision=revision,
                )
            else:
                raise
        
        t_config = time.time()
        print(f"Downloaded model configuration in {t_config - t_start:.2f} seconds")
        
        # Download tokenizer explicitly if requested or if skipping weights
        if download_tokenizer_only or skip_weights:
            print("Downloading tokenizer...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                tokenizer_path = os.path.join(output_dir, "tokenizer")
                os.makedirs(tokenizer_path, exist_ok=True)
                tokenizer.save_pretrained(tokenizer_path)
                print(f"Tokenizer saved to {tokenizer_path}")
            except Exception as e:
                print(f"Error downloading tokenizer: {e}")
                print("Continuing with other files...")
        
        # Download model weights if not skipping
        if not download_tokenizer_only and not skip_weights:
            print("Downloading model weights (this may take a while)...")
            try:
                model_kwargs = {}
                if low_memory:
                    model_kwargs.update({
                        "low_cpu_mem_usage": True,
                        "torch_dtype": "auto",
                    })
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    **model_kwargs
                )
                model_save_path = os.path.join(output_dir, "model")
                os.makedirs(model_save_path, exist_ok=True)
                model.save_pretrained(model_save_path)
                print(f"Model weights saved to {model_save_path}")
            except Exception as e:
                print(f"Error downloading model weights: {e}")
                print("You may need to download weights separately or try a smaller model")
        
        t_end = time.time()
        print(f"Download completed in {t_end - t_start:.2f} seconds")
        
        return output_dir
    
    except Exception as e:
        print(f"Error downloading model: {e}")
        
        # Provide troubleshooting help
        print("\nTroubleshooting suggestions:")
        print("1. Check your internet connection")
        print("2. Try increasing the timeout (--timeout)")
        print("3. Try a smaller model like 'distilgpt2' instead of GPT Neo")
        print("4. If on a corporate network, check if you need proxy settings:")
        print("   export HTTPS_PROXY=http://proxy-server:port")
        print("5. Try with --skip-weights to only download the model structure")
        
        return None
    finally:
        # Restore original environment variable if it existed
        if original_timeout is not None:
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = original_timeout
        else:
            os.environ.pop("HF_HUB_DOWNLOAD_TIMEOUT", None)

def download_simple_model():
    """Download a simpler, smaller model (distilgpt2) as an alternative"""
    model_id = "distilgpt2"
    output_dir = os.path.join("models", model_id)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading smaller model {model_id} to {output_dir}")
    try:
        # Download config and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer_path = os.path.join(output_dir, "tokenizer")
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
        
        # Download model
        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
        model_path = os.path.join(output_dir, "model")
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        print(f"Model saved to {model_path}")
        
        return output_dir
    except Exception as e:
        print(f"Error downloading smaller model: {e}")
        return None

def verify_download(output_dir):
    """Verify the model was downloaded correctly"""
    model_path = os.path.join(output_dir, "model")
    tokenizer_path = os.path.join(output_dir, "tokenizer")
    
    if os.path.exists(os.path.join(output_dir, "config.json")):
        print("✓ Model configuration found")
    else:
        print("✗ Model configuration not found")
    
    if os.path.exists(tokenizer_path) and os.listdir(tokenizer_path):
        print("✓ Tokenizer files found")
    else:
        print("✗ Tokenizer files not found")
    
    if os.path.exists(model_path) and os.listdir(model_path):
        print("✓ Model weights found")
    elif os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
        print("✓ Model weights found (in root directory)")
    else:
        print("✗ Model weights not found")

def main():
    args = parse_args()
    
    print("=" * 60)
    print("GPT NEO MODEL DOWNLOADER")
    print("=" * 60)
    
    # Set environment variables for extended timeouts
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.timeout)
    os.environ["TRANSFORMERS_REQUEST_TIMEOUT"] = str(args.timeout)
    
    # Download the model
    if args.smaller_model:
        output_dir = download_simple_model()
    else:
        output_dir = download_model_with_snapshot(
            model_id=args.model,
            output_dir=args.output,
            timeout=args.timeout,
            download_tokenizer_only=args.download_tokenizer_only,
            revision=args.revision,
            skip_weights=args.skip_weights,
            low_memory=args.low_memory
        )
    
    if output_dir:
        print("\nDownload Summary:")
        print("-" * 40)
        verify_download(output_dir)
        print("\nNext steps:")
        print(f"1. Use the model with: --local_model_path {output_dir}")
        print(f"2. Run analysis: python analyze_gptneo.py --local_model_path {output_dir} --text_file your_texts.txt")
    
if __name__ == "__main__":
    main()