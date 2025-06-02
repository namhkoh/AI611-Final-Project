#!/bin/bash

# DDPO VLM Comparison Environment Setup Script
# This script sets up the conda environment for the VLM comparison framework

echo "🔥 Setting up DDPO VLM Comparison Environment"
echo "=============================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check for CUDA availability
echo "🔍 Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    USE_CUDA=true
else
    echo "⚠️  No NVIDIA GPU detected. Installing CPU-only version."
    USE_CUDA=false
fi

# Create environment from YAML file
echo ""
echo "📦 Creating conda environment from environment.yml..."
if [ -f "environment.yml" ]; then
    conda env create -f environment.yml
    if [ $? -eq 0 ]; then
        echo "✅ Environment created successfully!"
    else
        echo "❌ Failed to create environment from YAML. Trying manual installation..."
        
        # Fallback: Manual installation
        echo "📦 Creating environment manually..."
        conda create -n ddpo-vlm python=3.10 -y
        conda activate ddpo-vlm
        
        if [ "$USE_CUDA" = true ]; then
            echo "Installing PyTorch with CUDA support..."
            conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        else
            echo "Installing PyTorch CPU-only..."
            conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        fi
        
        # Install other conda packages
        conda install numpy scipy pandas matplotlib seaborn jupyter ipykernel git ffmpeg -c conda-forge -y
        
        # Install pip packages
        pip install -r requirements.txt
    fi
else
    echo "❌ environment.yml not found. Creating environment manually..."
    
    # Manual installation
    echo "📦 Creating environment manually..."
    conda create -n ddpo-vlm python=3.10 -y
    
    echo "Activating environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ddpo-vlm
    
    if [ "$USE_CUDA" = true ]; then
        echo "Installing PyTorch with CUDA support..."
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        echo "Installing PyTorch CPU-only..."
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    fi
    
    # Install other conda packages
    conda install numpy scipy pandas matplotlib seaborn jupyter ipykernel git ffmpeg -c conda-forge -y
    
    # Install pip packages
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo "Installing core packages via pip..."
        pip install transformers>=4.37.0 accelerate>=0.25.0 diffusers>=0.26.0 ml-collections wandb
    fi
fi

# Install the package in development mode
echo ""
echo "📦 Installing ddpo-pytorch in development mode..."
pip install -e .

# Verify installation
echo ""
echo "🧪 Verifying installation..."
python -c "
import torch
import transformers
import diffusers
import accelerate
from ddpo_pytorch.vlm_comparison import VLMComparator

print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU count: {torch.cuda.device_count()}')
print(f'✅ Transformers version: {transformers.__version__}')
print(f'✅ Diffusers version: {diffusers.__version__}')
print(f'✅ Accelerate version: {accelerate.__version__}')
print('✅ VLM Comparator imported successfully')
print('')
print('🎉 Installation completed successfully!')
"

echo ""
echo "🚀 Setup completed! To activate the environment, run:"
echo "   conda activate ddpo-vlm"
echo ""
echo "🔥 Quick start commands:"
echo "   # Basic VLM comparison (CLIP only)"
echo "   python scripts/vlm_comparison_demo.py --models clip --test-robustness"
echo ""
echo "   # Auto-select models based on GPU memory"  
echo "   python scripts/vlm_comparison_demo.py --auto-select-models --test-robustness --test-efficiency"
echo ""
echo "   # Run DDPO training with CLIP"
echo "   accelerate launch scripts/train.py --config config/vlm_comparison.py:clip_similarity_experiment"
echo ""
echo "📚 See VLM_COMPARISON_README.md for detailed documentation!" 