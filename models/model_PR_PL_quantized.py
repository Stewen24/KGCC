import torch
import torch.nn as nn
from quantization.autoquant_utils import quantize_sequential
from quantization.base_quantized_classes import QuantizedActivation, FP32Acts
from quantization.base_quantized_model import QuantizedModel
from models.model_PR_PL import Domain_adaption_model
import os
from collections import OrderedDict

parameter={'hidden_1':64,'hidden_2':64,'num_of_class':3,'cluster_weight':2,'low_rank':32,'upper_threshold':0.9,'lower_threshold':0.5,'boost_type':'linear'}
class QuantizedFeatureExtractor(QuantizedActivation):
    def __init__(self, hidden_1, hidden_2, quant_setup=None, **quant_params):
        super().__init__(**quant_params)
        self.model = quantize_sequential(
            nn.Sequential(OrderedDict([
                ("conv1", nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1)),
                ("relu1", nn.ReLU()),
                ("conv2", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),),
                ("relu2", nn.ReLU()),
                ("pool1", nn.MaxPool2d(kernel_size=2, stride=2)),
                ("conv3", nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),),
                ("relu3", nn.ReLU()),
                ("pool2", nn.MaxPool2d(kernel_size=2, stride=2)),
                ("flatten", nn.Flatten()),
                ("fc1", nn.Linear(128 * 2 * 2, hidden_1)),
                ("relu4", nn.ReLU()),
                ("dropout", nn.Dropout(p=0.25)),
                ("fc2", nn.Linear(hidden_1, hidden_2)),
                ("relu5", nn.ReLU()),
            ])),
            **quant_params,
        )

        if quant_setup == "LSQ_paper":
            self.model[0].activation_quantizer = FP32Acts()
            self.model[0].weight_quantizer.quantizer.n_bits = 8
            self.model[-1].weight_quantizer.quantizer.n_bits = 8
            self.model[-1].activation_quantizer.quantizer.n_bits = 8
            for layer in self.model.modules():
                if isinstance(layer, QuantizedActivation):
                    layer.activation_quantizer = FP32Acts()

    def forward(self, x):
        return self.model(x)

class QuantizedDomainAdaptionModel(QuantizedModel):
    def __init__(self, hidden_1=32, hidden_2=32, hidden_3=32, hidden_4=32, num_of_class=3, low_rank=32, quant_setup=None, **quant_params):
        super().__init__((1, 5, 32, 32))  # 假设输入大小
        self.fea_extrator_f = QuantizedFeatureExtractor(hidden_1, hidden_2, quant_setup, **quant_params)
        self.fea_extrator_g = QuantizedFeatureExtractor(hidden_3, hidden_4, quant_setup, **quant_params)
        self.U = nn.Parameter(torch.randn(low_rank, hidden_2), requires_grad=True)
        self.V = nn.Parameter(torch.randn(low_rank, hidden_4), requires_grad=True)
        self.P = torch.randn(num_of_class, hidden_4)

    def forward(self, source, target, source_label):
        feature_source_f = self.fea_extrator_f(source)
        feature_target_f = self.fea_extrator_f(target)
        feature_source_g = self.fea_extrator_f(source)
        # 更新矩阵 P
        self.P = torch.matmul(
            torch.inverse(torch.diag(source_label.sum(axis=0)) + torch.eye(self.num_of_class).to(source.device)),
            torch.matmul(source_label.T, feature_source_g)
        )
        # 计算存储矩阵
        self.stored_mat = torch.matmul(self.V, self.P.T)
        # 预测
        source_predict = torch.matmul(torch.matmul(self.U, feature_source_f.T).T, self.stored_mat)
        target_predict = torch.matmul(torch.matmul(self.U, feature_target_f.T).T, self.stored_mat)
        return source_predict, target_predict


def pr_pl_quantized(pretrained=True, model_dir="", device="cuda", load_type="fp32", **qparams):
    # Initialize FP32 model
    fp_model = Domain_adaption_model(device=device)
    if pretrained and load_type == "fp32":
        # Load model from pretrained FP32 weights
        if os.path.exists(model_dir):
            print(f"Loading pretrained FP32 weights from {model_dir}")
            state_dict = torch.load(model_dir)
            fp_model.load_state_dict(state_dict)
        # Create quantized model based on FP32 model
        quant_model = QuantizedDomainAdaptionModel(**qparams)

    elif pretrained and load_type == "quantized":
        # Load pretrained QuantizedModel weights
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Quantized model directory {model_dir} does not exist.")
        print(f"Loading pretrained quantized weights from {model_dir}")
        state_dict = torch.load(model_dir)
        # Initialize quantized model and load weights
        quant_model = QuantizedDomainAdaptionModel(**qparams)
        quant_model.load_state_dict(state_dict, strict=False)

    else:
        raise ValueError("Invalid load_type specified. Use 'fp32' or 'quantized'.")

    return quant_model, fp_model
