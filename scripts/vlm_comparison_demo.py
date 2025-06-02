#!/usr/bin/env python3
"""
VLM Comparison Demo Script

This script demonstrates:
1. Loading and comparing different VLMs
2. Testing robustness to adversarial attacks
3. Running lightweight DDPO with VLM rewards

Usage:
    python scripts/vlm_comparison_demo.py --models clip blip2 --test-robustness --run-ddpo
"""

import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import time
import logging
from pathlib import Path
import json

# Add the parent directory to the path to import ddpo_pytorch
import sys
sys.path.append(str(Path(__file__).parent.parent))

from ddpo_pytorch.vlm_comparison import VLMComparator
from ddpo_pytorch import rewards
import config.vlm_comparison as vlm_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_images():
    """Create simple test images for evaluation"""
    def create_test_image(text, color, size=(256, 256)):
        img = Image.new('RGB', size, color=color)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Calculate text position (center)
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = len(text) * 10, 20
        
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        draw.text((x, y), text, fill='white', font=font)
        return img
    
    return [
        (create_test_image("CAT", "orange"), "a cat"),
        (create_test_image("DOG", "brown"), "a dog"),
        (create_test_image("BIRD", "blue"), "a bird"),
        (create_test_image("FISH", "cyan"), "a fish")
    ]


def test_vlm_basic_functionality(comparator, models_to_test, test_data):
    """Test basic VLM functionality"""
    logger.info("🧪 Testing basic VLM functionality")
    
    results = {}
    for model_name in models_to_test:
        if model_name not in comparator.models:
            continue
            
        logger.info(f"Testing {model_name}...")
        model_results = []
        
        for i, (image, prompt) in enumerate(test_data):
            try:
                if model_name == "clip":
                    response = comparator.generate_response(model_name, image, prompt)
                else:
                    response = comparator.generate_response(model_name, image, "What is in this image?")
                
                logger.info(f"  Image {i+1} ({prompt}): {response[:50]}...")
                model_results.append(response)
            except Exception as e:
                logger.error(f"Error with {model_name} on image {i+1}: {e}")
                model_results.append(f"Error: {e}")
        
        results[model_name] = model_results
    
    return results


def test_vlm_robustness(comparator, models_to_test, test_image, test_prompt):
    """Test VLM robustness to different attacks"""
    logger.info("🛡️ Testing VLM robustness")
    
    attack_types = ["typographic", "adversarial_patch", "corruption"]
    robustness_results = {}
    
    for model_name in models_to_test:
        if model_name not in comparator.models:
            continue
            
        logger.info(f"Testing {model_name} robustness...")
        model_robustness = {}
        
        for attack in attack_types:
            try:
                result = comparator.evaluate_robustness(model_name, test_image, test_prompt, attack)
                score = result["robustness_score"]
                model_robustness[attack] = score
                logger.info(f"  {attack}: {score:.3f}")
            except Exception as e:
                logger.error(f"  {attack}: Error - {e}")
                model_robustness[attack] = 0.0
        
        robustness_results[model_name] = model_robustness
    
    return robustness_results


def benchmark_efficiency(comparator, models_to_test, test_data):
    """Benchmark VLM efficiency"""
    logger.info("⚡ Benchmarking efficiency")
    
    efficiency_results = {}
    
    for model_name in models_to_test:
        if model_name not in comparator.models:
            continue
            
        logger.info(f"Benchmarking {model_name}...")
        
        # Measure inference time
        times = []
        for image, prompt in test_data[:2]:  # Use subset for speed
            start_time = time.time()
            try:
                _ = comparator.generate_response(model_name, image, prompt)
                times.append(time.time() - start_time)
            except Exception as e:
                logger.error(f"Error during benchmark: {e}")
                times.append(float('inf'))
        
        avg_time = np.mean(times)
        memory_usage = comparator.model_info[model_name]["memory_gb"]
        
        efficiency_results[model_name] = {
            "avg_inference_time": avg_time,
            "memory_usage_gb": memory_usage
        }
        
        logger.info(f"  Average inference time: {avg_time:.3f}s")
        logger.info(f"  Memory usage: {memory_usage}GB")
    
    return efficiency_results


def test_reward_functions(models_to_test, test_data):
    """Test VLM-based reward functions"""
    logger.info("🎯 Testing reward functions")
    
    # Prepare test data
    test_images = np.array([np.array(img) for img, _ in test_data])
    test_prompts = [prompt for _, prompt in test_data]
    
    reward_results = {}
    
    for model_name in models_to_test:
        try:
            if model_name == "clip":
                reward_fn = rewards.clip_similarity()
                reward_name = "CLIP Similarity"
            elif model_name == "blip2":
                reward_fn = rewards.blip2_alignment()
                reward_name = "BLIP-2 Alignment"
            elif model_name == "instructblip":
                reward_fn = rewards.instructblip_reward()
                reward_name = "InstructBLIP"
            else:
                continue
            
            logger.info(f"Testing {reward_name}...")
            scores, info = reward_fn(test_images, test_prompts, [{}] * len(test_prompts))
            
            reward_results[model_name] = {
                "scores": scores.tolist(),
                "prompts": test_prompts,
                "info": info
            }
            
            for prompt, score in zip(test_prompts, scores):
                logger.info(f"  {prompt}: {score:.4f}")
                
        except Exception as e:
            logger.error(f"Error testing {model_name} reward function: {e}")
    
    return reward_results


def run_lightweight_ddpo(demo_model, epochs=3):
    """Run a lightweight DDPO demo"""
    logger.info("🚀 Running lightweight DDPO demo")
    
    if not demo_model:
        logger.warning("No model available for DDPO demo")
        return
    
    try:
        # Get lightweight config
        config = vlm_config.lightweight_demo()
        config.num_epochs = epochs
        
        if demo_model == "clip":
            config.reward_fn = "clip_similarity"
        elif demo_model == "blip2":
            config.reward_fn = "blip2_alignment"
        else:
            config.reward_fn = "aesthetic_score"  # fallback
        
        logger.info(f"DDPO Configuration:")
        logger.info(f"  Model: {config.pretrained.model}")
        logger.info(f"  Epochs: {config.num_epochs}")
        logger.info(f"  Reward function: {config.reward_fn}")
        logger.info(f"  Batch size: {config.sample.batch_size}")
        
        # Note: In a real implementation, you would call the training script here
        # For this demo, we just show the configuration
        logger.info("✅ DDPO configuration ready (use with scripts/train.py)")
        
        return config
        
    except Exception as e:
        logger.error(f"Error setting up DDPO: {e}")
        return None


def save_results(results, output_dir="./vlm_comparison_results"):
    """Save comparison results"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as JSON
    json_path = output_path / "comparison_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="VLM Comparison Demo")
    parser.add_argument("--models", nargs="+", 
                       choices=["clip", "blip2", "instructblip", "llava-1.5"],
                       default=["clip"],
                       help="Models to test")
    parser.add_argument("--test-robustness", action="store_true",
                       help="Test robustness to attacks")
    parser.add_argument("--test-efficiency", action="store_true", 
                       help="Benchmark efficiency")
    parser.add_argument("--test-rewards", action="store_true",
                       help="Test reward functions")
    parser.add_argument("--run-ddpo", action="store_true",
                       help="Run lightweight DDPO demo")
    parser.add_argument("--ddpo-epochs", type=int, default=3,
                       help="Number of DDPO epochs for demo")
    parser.add_argument("--output-dir", default="./vlm_comparison_results",
                       help="Output directory for results")
    parser.add_argument("--auto-select-models", action="store_true",
                       help="Automatically select models based on available GPU memory")
    
    args = parser.parse_args()
    
    # Initialize
    logger.info("🔥 VLM Comparison Demo Starting")
    comparator = VLMComparator()
    
    # Auto-select models based on GPU memory if requested
    if args.auto_select_models and torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Available GPU Memory: {total_memory:.1f} GB")
        
        if total_memory >= 20:
            args.models = ["clip", "blip2", "instructblip"]
            logger.info("🚀 Auto-selected: all models (high memory)")
        elif total_memory >= 15:
            args.models = ["clip", "blip2"]
            logger.info("⚡ Auto-selected: efficient models (medium memory)")
        else:
            args.models = ["clip"]
            logger.info("💡 Auto-selected: CLIP only (low memory)")
    
    # Load models
    logger.info(f"Loading models: {args.models}")
    for model_name in args.models:
        try:
            logger.info(f"Loading {model_name}...")
            comparator.load_model(model_name)
            logger.info(f"✅ {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            args.models.remove(model_name)
    
    if not args.models:
        logger.error("No models loaded successfully. Exiting.")
        return
    
    # Create test data
    test_data = create_test_images()
    logger.info(f"Created {len(test_data)} test images")
    
    # Initialize results
    all_results = {
        "models_tested": args.models,
        "model_info": comparator.get_model_comparison()
    }
    
    # Print model comparison table
    comparator.print_comparison_table()
    
    # Test basic functionality
    basic_results = test_vlm_basic_functionality(comparator, args.models, test_data)
    all_results["basic_functionality"] = basic_results
    
    # Test robustness
    if args.test_robustness:
        test_image, test_prompt = test_data[0]
        robustness_results = test_vlm_robustness(comparator, args.models, test_image, test_prompt)
        all_results["robustness"] = robustness_results
    
    # Test efficiency
    if args.test_efficiency:
        efficiency_results = benchmark_efficiency(comparator, args.models, test_data)
        all_results["efficiency"] = efficiency_results
    
    # Test reward functions
    if args.test_rewards:
        reward_results = test_reward_functions(args.models, test_data)
        all_results["reward_functions"] = reward_results
    
    # Run DDPO demo
    if args.run_ddpo:
        demo_model = args.models[0] if args.models else None
        ddpo_config = run_lightweight_ddpo(demo_model, args.ddpo_epochs)
        all_results["ddpo_demo"] = {
            "model_used": demo_model,
            "config": str(ddpo_config) if ddpo_config else None
        }
    
    # Save results
    save_results(all_results, args.output_dir)
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 DEMO SUMMARY")
    logger.info("="*50)
    logger.info(f"Models tested: {', '.join(args.models)}")
    logger.info(f"Tests completed:")
    logger.info(f"  ✅ Basic functionality")
    if args.test_robustness:
        logger.info(f"  ✅ Robustness analysis")
    if args.test_efficiency:
        logger.info(f"  ✅ Efficiency benchmarking")
    if args.test_rewards:
        logger.info(f"  ✅ Reward function testing")
    if args.run_ddpo:
        logger.info(f"  ✅ DDPO demo setup")
    
    logger.info(f"\n🎉 Demo completed! Results saved to {args.output_dir}")
    logger.info("🚀 Next steps:")
    logger.info("  1. Review the comparison results")
    logger.info("  2. Run full DDPO training with: accelerate launch scripts/train.py --config config/vlm_comparison.py:clip_similarity_experiment")
    logger.info("  3. Explore the Colab notebook for interactive analysis")


if __name__ == "__main__":
    main() 