#!/usr/bin/env python3
"""
Experiment Log Updater

This script automatically updates the experiment log markdown file
with results from VLM comparison experiments.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def load_experiment_results(results_path: str = "vlm_comparison_results/comparison_results.json") -> Dict[str, Any]:
    """Load experiment results from JSON file"""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_path}")
        return {}
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in results file: {results_path}")
        return {}


def update_basic_functionality_table(log_content: str, results: Dict[str, Any]) -> str:
    """Update the basic functionality test table"""
    if "basic_functionality" not in results:
        return log_content
    
    # Extract responses for each model
    basic_results = results["basic_functionality"]
    
    for model_name, responses in basic_results.items():
        if model_name in ["clip"]:
            # For CLIP, extract similarity scores
            scores = []
            for response in responses:
                if "Similarity score:" in response:
                    score = float(response.split(":")[1].strip())
                    scores.append(f"{score:.4f}")
                else:
                    scores.append("N/A")
            
            # Update the table row for this model
            pattern = rf"\| \*\*{model_name.upper()}\*\* \| \[PENDING\] \| \[PENDING\] \| \[PENDING\] \| \[PENDING\] \| \[PENDING\] \|"
            if len(scores) >= 4:
                avg_score = sum(float(s) for s in scores[:4] if s != "N/A") / len([s for s in scores[:4] if s != "N/A"])
                replacement = f"| **{model_name.upper()}** | {scores[0]} | {scores[1]} | {scores[2]} | {scores[3]} | {avg_score:.4f} |"
                log_content = re.sub(pattern, replacement, log_content)
    
    return log_content


def update_robustness_table(log_content: str, results: Dict[str, Any]) -> str:
    """Update the robustness analysis table"""
    if "robustness" not in results:
        return log_content
    
    robustness_results = results["robustness"]
    
    for model_name, attacks in robustness_results.items():
        pattern = rf"\| \*\*{model_name.upper()}\*\* \| \[PENDING\] \| \[PENDING\] \| \[PENDING\] \| \[PENDING\] \|"
        
        typo_score = attacks.get("typographic", 0)
        patch_score = attacks.get("adversarial_patch", 0)
        corr_score = attacks.get("corruption", 0)
        avg_score = (typo_score + patch_score + corr_score) / 3
        
        replacement = f"| **{model_name.upper()}** | {typo_score:.3f} | {patch_score:.3f} | {corr_score:.3f} | {avg_score:.3f} |"
        log_content = re.sub(pattern, replacement, log_content)
    
    return log_content


def update_efficiency_table(log_content: str, results: Dict[str, Any]) -> str:
    """Update the efficiency benchmarking table"""
    if "efficiency" not in results:
        return log_content
    
    efficiency_results = results["efficiency"]
    
    for model_name, metrics in efficiency_results.items():
        pattern = rf"\| \*\*{model_name.upper()}\*\* \| \[PENDING\] \| \d+ \| \[PENDING\] \| \[PENDING\] \|"
        
        avg_time = metrics.get("avg_inference_time", 0)
        memory_gb = metrics.get("memory_usage_gb", 0)
        throughput = 1 / avg_time if avg_time > 0 else 0
        efficiency_score = throughput / memory_gb if memory_gb > 0 else 0
        
        replacement = f"| **{model_name.upper()}** | {avg_time:.3f} | {memory_gb} | {throughput:.2f} | {efficiency_score:.3f} |"
        log_content = re.sub(pattern, replacement, log_content)
    
    return log_content


def update_model_loading_status(log_content: str, results: Dict[str, Any]) -> str:
    """Update model loading status"""
    if "models_tested" not in results:
        return log_content
    
    models_tested = results["models_tested"]
    
    # Update status for each model
    for model in ["blip2", "instructblip"]:
        if model in models_tested:
            pattern = rf"- \*\*{model.upper()}:\*\* \[PENDING\]"
            replacement = f"- **{model.upper()}:** ✅ Loaded successfully"
            log_content = re.sub(pattern, replacement, log_content)
    
    return log_content


def update_experiment_status(log_content: str, status: str = "COMPLETED") -> str:
    """Update experiment status and timestamp"""
    # Update status
    log_content = re.sub(
        r"\*\*Experiment Status:\*\* 🔄 IN PROGRESS",
        f"**Experiment Status:** ✅ {status}",
        log_content
    )
    
    # Update timestamp
    current_time = datetime.now().strftime("%B %d, %Y %H:%M")
    log_content = re.sub(
        r"\*\*Last Updated:\*\* June 2, 2025",
        f"**Last Updated:** {current_time}",
        log_content
    )
    
    return log_content


def generate_conclusions(results: Dict[str, Any]) -> str:
    """Generate conclusions based on results"""
    if not results:
        return "[NO RESULTS TO ANALYZE]"
    
    conclusions = []
    
    # Analyze efficiency
    if "efficiency" in results:
        fastest_model = min(results["efficiency"].items(), 
                          key=lambda x: x[1].get("avg_inference_time", float('inf')))
        conclusions.append(f"🚀 **Fastest Model:** {fastest_model[0].upper()} ({fastest_model[1]['avg_inference_time']:.3f}s)")
    
    # Analyze robustness
    if "robustness" in results:
        robustness_scores = {}
        for model, attacks in results["robustness"].items():
            avg_robustness = sum(attacks.values()) / len(attacks)
            robustness_scores[model] = avg_robustness
        
        most_robust = max(robustness_scores.items(), key=lambda x: x[1])
        conclusions.append(f"🛡️ **Most Robust:** {most_robust[0].upper()} ({most_robust[1]:.3f})")
    
    # Memory efficiency
    if "model_info" in results:
        memory_efficient = min(results["model_info"].items(), 
                             key=lambda x: x[1].get("memory_gb", float('inf')))
        conclusions.append(f"💾 **Most Memory Efficient:** {memory_efficient[0].upper()} ({memory_efficient[1]['memory_gb']}GB)")
    
    return "\n".join(f"- {conclusion}" for conclusion in conclusions)


def update_experiment_log(log_path: str = "experiments/experiment_log.md", 
                         results_path: str = "vlm_comparison_results/comparison_results.json"):
    """Main function to update the experiment log"""
    
    print("📊 Updating experiment log...")
    
    # Load results
    results = load_experiment_results(results_path)
    if not results:
        print("❌ No results to update")
        return
    
    # Read current log
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"❌ Log file not found: {log_path}")
        return
    
    # Update different sections
    log_content = update_basic_functionality_table(log_content, results)
    log_content = update_robustness_table(log_content, results)
    log_content = update_efficiency_table(log_content, results)
    log_content = update_model_loading_status(log_content, results)
    
    # Update conclusions
    conclusions = generate_conclusions(results)
    log_content = re.sub(
        r"\[TO BE FILLED AFTER EXPERIMENT COMPLETION\]",
        conclusions,
        log_content
    )
    
    # Update status
    log_content = update_experiment_status(log_content, "COMPLETED")
    
    # Save updated log
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(log_content)
    
    print(f"✅ Experiment log updated: {log_path}")
    print("📋 Key updates:")
    print("  - Basic functionality scores")
    print("  - Robustness analysis results") 
    print("  - Efficiency benchmarks")
    print("  - Model loading status")
    print("  - Experiment conclusions")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update experiment log with results")
    parser.add_argument("--log", default="experiments/experiment_log.md", 
                       help="Path to experiment log markdown file")
    parser.add_argument("--results", default="vlm_comparison_results/comparison_results.json",
                       help="Path to results JSON file") 
    
    args = parser.parse_args()
    
    update_experiment_log(args.log, args.results) 