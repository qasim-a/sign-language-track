"""
convert_to_onnx.py
Converts the trained PyTorch .pth model to ONNX format for browser deployment.
Run from project root: python src/convert_to_onnx.py
Output: models/asl_model.onnx
"""

import torch
import torch.nn as nn

SIGNS = [
    "hello", "yes", "no", "nothing", "thank you", "please", "eat", "drink",
    "water", "more", "apple", "mother", "father", "book", "walk", "cold",
    "hot", "black", "carrot", "go", "day", "break", "cow", "monkey", "draw", "type"
]

class ASLModel(nn.Module):
    def __init__(self):
        super(ASLModel, self).__init__()
        self.lstm = nn.LSTM(input_size=225, hidden_size=128, num_layers=2,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, len(SIGNS))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# load the saved model weights
model = ASLModel()
model.load_state_dict(torch.load("models/asl_model_25signs_85pct.pth", map_location="cpu"))
model.eval()

# dummy input matching sequence shape: (batch=1, frames=30, features=225)
dummy_input = torch.randn(1, 30, 225)

torch.onnx.export(
    model,
    dummy_input,
    "models/asl_model.onnx",
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"Exported successfully to models/asl_model.onnx")
print(f"Signs ({len(SIGNS)}): {', '.join(SIGNS)}")