import torch
import torch.nn as nn
from torchvision import models

def export_effnet_stable(model_path, output_name, num_classes):
    print(f"--- Exporting {output_name} ---")
    model = models.efficientnet_b2()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    # SUCCESS TACTIC:
    # 1. Use opset_version=18 (Matches your PyTorch 2.9 native version)
    # 2. Add export_modules_as_functions=False for a flatter graph
    torch.onnx.export(
        model, 
        dummy_input, 
        output_name,
        export_params=True,
        opset_version=18,  
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # We simplify dynamic axes for B2 models to save memory
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ Clean Export: {output_name}")

export_effnet_stable(r"D:\Hackathon\Tech Sprint\eye_effnet_best (1).pt", "eye_freshness.onnx", 3)
export_effnet_stable(r"D:\Hackathon\Tech Sprint\gill_model.pt", "gill_freshness.onnx", 2)