"""Configuration for multi-device support across NVIDIA GPU, Mac MPS, and CPU."""
import torch

def get_device_info():
    """Print available device information."""
    print("\n" + "="*60)
    print("DEVICE INFORMATION")
    print("="*60)
    
    print(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA Available: Yes")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"✓ CUDA Available: No")
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"✓ Metal Performance Shaders: Yes")
    else:
        print(f"✓ Metal Performance Shaders: No")
    
    print(f"✓ CPU Cores: {torch.get_num_threads()}")
    print("="*60 + "\n")

def setup_device(device_str: str = None):
    """Setup and return the best available device.
    
    Args:
        device_str: Explicitly specify device ('cuda', 'mps', 'cpu', or None for auto)
    
    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    if device_str:
        if device_str == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Using CPU.")
            return "cpu"
        elif device_str == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("Warning: MPS requested but not available. Using CPU.")
            return "cpu"
        return device_str
    
    # Auto-detect best available device
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

if __name__ == "__main__":
    get_device_info()
    device = setup_device()
    print(f"Selected Device: {device}")
