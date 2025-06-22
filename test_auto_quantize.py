"""
🏀 Test Auto-Selection of Quantization Methods

Demonstrates how HoopGT automatically chooses the best quantization method.
"""

import torch
import torch.nn as nn
from hoopgt import HoopQuantizer


class CNNModel(nn.Module):
    """CNN model - should prefer static quantization"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(1)
        return self.fc(x)


class RNNModel(nn.Module):
    """RNN model - should prefer dynamic quantization"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(100, 50, batch_first=True)
        self.fc = nn.Linear(50, 10)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class SimpleModel(nn.Module):
    """Simple Linear model - should default to dynamic"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


def test_auto_quantization_selection():
    """Test automatic quantization method selection"""
    
    quantizer = HoopQuantizer()
    target = "apple-silicon"
    
    # Test CNN model
    cnn_model = CNNModel()
    cnn_method = quantizer.get_recommended_method(cnn_model, target)
    print(f"🖼️  CNN Model → {cnn_method} quantization")
    
    # Test RNN model  
    rnn_model = RNNModel()
    rnn_method = quantizer.get_recommended_method(rnn_model, target)
    print(f"🔄 RNN Model → {rnn_method} quantization")
    
    # Test simple model
    simple_model = SimpleModel()
    simple_method = quantizer.get_recommended_method(simple_model, target)
    print(f"➡️  Simple Model → {simple_method} quantization")
    
    # Verify expectations
    assert cnn_method == "static", f"CNN should use static, got {cnn_method}"
    assert rnn_method == "dynamic", f"RNN should use dynamic, got {rnn_method}"
    assert simple_method == "dynamic", f"Simple should use dynamic, got {simple_method}"
    
    print("✅ All auto-selection tests passed!")
    return True


if __name__ == "__main__":
    print("🧪 Testing automatic quantization method selection...")
    test_auto_quantization_selection()
    print("🎉 Auto-selection works perfectly!")
    print("\n💡 Users can simply use --quantize and HoopGT will choose the best method!") 