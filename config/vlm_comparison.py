"""
Configuration for VLM Comparison Experiments

This config file contains different experimental setups for comparing VLMs
including robustness analysis and efficiency benchmarks.
"""

import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def blip2_experiment():
    """BLIP-2 alignment experiment - more efficient than LLaVA"""
    config = base.get_config()
    
    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    config.num_epochs = 50
    config.use_lora = True
    config.save_freq = 10
    
    # Smaller batch sizes for free-tier GPU compatibility
    config.sample.batch_size = 2
    config.sample.num_batches_per_epoch = 2
    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 2
    
    # Use simple animals for easier evaluation
    config.prompt_fn = "simple_animals"
    config.prompt_fn_kwargs = {}
    
    # Use BLIP-2 reward function
    config.reward_fn = "blip2_alignment"
    
    config.per_prompt_stat_tracking = {
        "buffer_size": 8,
        "min_count": 4,
    }
    
    return config


def clip_similarity_experiment():
    """CLIP similarity experiment - most efficient"""
    config = base.get_config()
    
    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    config.num_epochs = 30
    config.use_lora = True
    config.save_freq = 10
    
    # Very small batch sizes for maximum compatibility
    config.sample.batch_size = 1
    config.sample.num_batches_per_epoch = 4
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 4
    
    config.prompt_fn = "simple_animals"
    config.prompt_fn_kwargs = {}
    
    # Use CLIP reward function
    config.reward_fn = "clip_similarity"
    
    config.per_prompt_stat_tracking = {
        "buffer_size": 8,
        "min_count": 4,
    }
    
    return config


def instructblip_experiment():
    """InstructBLIP experiment - balanced performance"""
    config = base.get_config()
    
    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    config.num_epochs = 40
    config.use_lora = True
    config.save_freq = 10
    
    config.sample.batch_size = 2
    config.sample.num_batches_per_epoch = 2
    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 2
    
    config.prompt_fn = "simple_animals"
    config.prompt_fn_kwargs = {}
    
    # Use InstructBLIP reward function
    config.reward_fn = "instructblip_reward"
    
    config.per_prompt_stat_tracking = {
        "buffer_size": 8,
        "min_count": 4,
    }
    
    return config


def robustness_experiment():
    """Experiment focused on evaluating robustness to attacks"""
    config = base.get_config()
    
    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    config.num_epochs = 25
    config.use_lora = True
    config.save_freq = 5
    
    config.sample.batch_size = 1
    config.sample.num_batches_per_epoch = 3
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 3
    
    config.prompt_fn = "simple_animals"
    config.prompt_fn_kwargs = {}
    
    # Use robustness-focused reward
    config.reward_fn = "vlm_robustness_reward"
    config.reward_fn_kwargs = {
        "model_name": "blip2",
        "attack_type": "typographic"
    }
    
    config.per_prompt_stat_tracking = {
        "buffer_size": 6,
        "min_count": 3,
    }
    
    return config


def lightweight_demo():
    """Lightweight demo configuration for Colab/free-tier GPUs"""
    config = base.get_config()
    
    # Use smaller SD model for reduced memory
    config.pretrained.model = "CompVis/stable-diffusion-v1-4"
    config.num_epochs = 10
    config.use_lora = True
    config.save_freq = 5
    config.mixed_precision = "fp16"
    
    # Minimal batch sizes
    config.sample.batch_size = 1
    config.sample.num_batches_per_epoch = 2
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 2
    
    # Simple prompts
    config.prompt_fn = "simple_animals"
    config.prompt_fn_kwargs = {}
    
    # Use most efficient reward (CLIP)
    config.reward_fn = "clip_similarity"
    
    config.per_prompt_stat_tracking = {
        "buffer_size": 4,
        "min_count": 2,
    }
    
    return config


def get_config(name):
    return globals()[name]() 