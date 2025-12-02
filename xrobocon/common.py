import os

# 1. MPS Fallback Setting
# This must be set before torch is imported (which happens in stable_baselines3)
if 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ:
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import genesis as gs
from stable_baselines3 import PPO

def setup_genesis(backend=gs.gpu):
    """
    Safely initialize Genesis.
    Checks if it's already initialized to avoid errors.
    """
    # Check if is_initialized exists (for newer versions)
    if hasattr(gs, 'is_initialized'):
        if not gs.is_initialized():
            gs.init(backend=backend)
    else:
        # Fallback for older versions or if method doesn't exist
        # We try to init, catching the specific error if possible, 
        # but since we can't easily check state, we assume the caller 
        # calls this once or we rely on the fact that repeated calls might error.
        # However, based on user logs, repeated calls cause errors.
        # A simple workaround for this specific environment/version issue:
        try:
            gs.init(backend=backend)
        except Exception as e:
            # If it says "already initialized", we ignore it.
            if "already initialized" in str(e):
                pass
            else:
                # For versions where we can't check, we might just have to risk it 
                # or rely on the script structure.
                # But let's try to be safe.
                print(f"Genesis init warning: {e}")

def load_trained_model(model_path, env):
    """
    Load a trained PPO model safely.
    """
    # SB3 automatically adds .zip, so we should check for that too
    if not os.path.exists(model_path):
        if os.path.exists(model_path + ".zip"):
            model_path += ".zip"
        else:
            raise FileNotFoundError(f"Model file not found: {model_path} (or {model_path}.zip)")
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, env=env)
    return model
