import numpy as np
from torch import nn
from backbones.atrfas.res_u_net import ResUNet
from backbones.atrfas.easy_res_u_net import EasyResUNet
from backbones.atrfas.base_layer import DownConvNormAct, ConvNormAct, Reshape, L2Normalize, Mean
import torch
import cv2
from PIL import Image
from torchvision import transforms


# type gating nerwork
class TyepGatingNetwork(nn.Module):
    def __init__(self, number_attacks=3):
        super(TyepGatingNetwork, self).__init__()
        self.gate = nn.Sequential(
            DownConvNormAct(3, 32),  # [1, 3, 6, 256, 256] -> [1, 32, 6, 128, 128]
            DownConvNormAct(32, 64),  # [1, 32, 6, 128, 128] -> [1, 64, 6, 64, 64]
            DownConvNormAct(64, 64, kernel_size=7),  # [1, 64, 6, 64, 64] -> [1, 64, 6, 32, 32]
            torch.nn.AdaptiveAvgPool3d((1, 1, 1)),  # [1, 64, 6, 32, 32] -> [1, 64, 6, 1, 1]
            Reshape(64),  # [1, 64, 1, 1, 1] -> [1, 64]
            nn.Linear(64, 32),  # [1, 32]
            nn.Linear(32, number_attacks),  # [1, 3]
        )

    def forward(self, x):
        assert x.dim() == 5, "input tensor must be 5D, but got {}D".format(x.dim())
        # assert x.shape[2]==6, "input tensor must have 6 frames, but got {}".format(x) #zzzz
        assert x.shape[1] == 3, "input tensor must have 3 channels, but got {}".format(x)

        x = self.gate(x)  # [B, C, N, H, W] -> [B, M]
        x = torch.softmax(x, dim=-1)
        return x


# MEMM
class MEMM(nn.Module):
    def __init__(self, num_frames: int = 6):
        super(MEMM, self).__init__()
        self.head_stem = nn.Sequential(
            DownConvNormAct(3, 32),  # [6, 32, 128, 128]
            DownConvNormAct(32, 64),  # [6, 64, 64, 64]
        )
        self.positional_embedding = torch.nn.Parameter(torch.randn(1, 64, num_frames, 64, 64))  # [B, 64, N, 64, 64]
        depth_map_cor = np.reshape(np.arange(256) / 255., [1, 1, 1, 1, -1]).astype(np.float32)
        self.register_buffer('depth_map_cor', torch.from_numpy(depth_map_cor))

        self.resunet = ResUNet() #zzzzz
        # self.resunet0 = ResUNet()
        # self.resunet1 = ResUNet()
        # self.resunet2 = ResUNet()

    @staticmethod
    def pixel_wise_softmax(x):
        """
        Applies a pixel-wise softmax operation to the input tensor.

        The function moves the channel dimension to the last position, computes the
        exponential of each element subtracted by the maximum value in its channel
        (for numerical stability), and normalizes by the sum of exponentials along
        the channel dimension.
        """
        # Move the channel dimension to the last
        x = x.permute(0, 2, 3, 4, 1)
        channel_max, _ = torch.max(x, dim=4, keepdim=True)
        exponential_map = torch.exp(x - channel_max)
        normalize = torch.sum(exponential_map, dim=4, keepdims=True)
        return exponential_map / (normalize + 1e-5)

    def forward(self, x: torch.tensor):
        assert x.dim() == 5, "input tensor must be 5D, but got {}D".format(x.dim())
        # assert x.shape[2]==6, "input tensor must have 6 frames, but got {}".format(x) #zzzz
        assert x.shape[1] == 3, "input tensor must have 3 channels, but got {}".format(x)

        # input embedding
        x = self.head_stem(x)  # [B, C, N, H, W] -> [B, 64, N, 64, 64]
        # x = x + self.positional_embedding #??????
        # number of types
        # M = type_gating.shape[1]

        # result from
        # x_bar = [self.resunet(x) for _ in range(M)] #zzzz
        # x_bar = [self.resunet0(x), self.resunet1(x), self.resunet2(x)]
        # type_gating = torch.reshape(type_gating, [-1, 3, 1, 1, 1])  # [B, M] -> [M, B, 1, 1, 1]
        x_prime = self.resunet(x)

        # x_prime = sum(x_bar[i] * type_gating[:, i:i + 1, :, :, :] for i in range(M))
        depth_softmax = MEMM.pixel_wise_softmax(x_prime)

        depth_map = torch.sum(depth_softmax * self.depth_map_cor, axis=-1)

        return depth_map


# attention gating network
class AttentionGatingNetwork(nn.Module):
    def __init__(self):
        super(AttentionGatingNetwork, self).__init__()
        self.attention_stem = nn.Sequential(
            DownConvNormAct(3, 16),
            DownConvNormAct(16, 32),
        )
        self.easyunet = EasyResUNet()
        depth_map_cor = np.reshape(np.arange(256) / 255., [1, 1, 1, 1, -1]).astype(np.float32)
        self.register_buffer('depth_map_cor', torch.from_numpy(depth_map_cor))

    @staticmethod
    def pixel_wise_softmax(x):
        """
        Applies a pixel-wise softmax operation to the input tensor.

        The function moves the channel dimension to the last position, computes the
        exponential of each element subtracted by the maximum value in its channel
        (for numerical stability), and normalizes by the sum of exponentials along
        the channel dimension.
        """
        # Move the channel dimension to the last
        x = x.permute(0, 2, 3, 4, 1)
        channel_max, _ = torch.max(x, dim=3, keepdim=True)
        exponential_map = torch.exp(x - channel_max)
        normalize = torch.sum(exponential_map, dim=3, keepdims=True)
        return exponential_map / (normalize + 1e-5)

    def forward(self, x: torch.tensor):
        assert x.dim() == 5, "input tensor must be 5D, but got {}D".format(x.dim())
        # assert x.shape[2]==6, "input tensor must have 6 frames, but got {}".format(x) #zzzzz
        assert x.shape[1] == 3, "input tensor must have 3 channels, but got {}".format(x)

        atten_x = self.attention_stem(x)  # convert from (N, C, H, W) to (N, 32, 64, 64)
        attention_x = self.easyunet(atten_x)  # convert from (N, 32)

        attention_soft_max = AttentionGatingNetwork.pixel_wise_softmax(attention_x)

        dot = self.depth_map_cor * attention_soft_max
        summation = torch.sum(dot, dim=-1)
        attention_map = torch.unsqueeze(summation, dim=1)
        # attention_map = torch.softmax(torch.reshape(attention_map, [attention_map.shape[0],6, 64, 64]), dim=1) # zzzzzz
        attention_map = torch.softmax(torch.reshape(attention_map, [attention_map.shape[0], 1, 64, 64]), dim=1)
        return attention_map


# classification head
class ClassificationHead(nn.Module):
    def __init__(self, return_proba=False, num_class=2):
        super(ClassificationHead, self).__init__()
        self.return_proba = return_proba
        self.f_net = nn.Sequential(
            DownConvNormAct(1, 16),  # 32*32*16
            ConvNormAct(16, 8, 3),  # 32*32*8
            ConvNormAct(8, 4, 3),  # 32*32*4
            Reshape(32 * 32 * 4),
            L2Normalize(1),
            nn.Dropout(0.5),
            nn.Linear(32 * 32 * 4, num_class)
        )

    def forward(self, x: torch.tensor):
        assert x.dim() == 3, "input tensor must be 3D, but got {}D".format(x.dim())

        x_final = x.unsqueeze(1).unsqueeze(2)
        pred_logits = self.f_net(x_final)
        if self.return_proba:
            return torch.softmax(pred_logits, dim=-1)
        return pred_logits


class ATRFAS(nn.Module):
    def __init__(self, num_class=2, num_frames: int = 6, return_proba: bool = False, infer_type: str = 'inference'):
        super(ATRFAS, self).__init__()
        # self.type_gating_network = TyepGatingNetwork()
        self.memm = MEMM(num_frames)
        self.attention_gating_network = AttentionGatingNetwork()
        self.classification_head = ClassificationHead(return_proba, num_class)

        self.infer_type = infer_type

    def forward(self, x: torch.tensor):
        # assert x.dim() == 5, "input tensor must be 5D, but got {}D".format(x.dim()) #zzzz
        assert x.shape[2] == 3, "input tensor must have 6 frames, but got {}".format(x)

        x = x.permute(0, 2, 1, 3, 4)  # [B, N, C, H, W] -> [B, C, N, H, W]

        # type_gating = self.type_gating_network(x)
        # frame_depth_map = self.memm(x, type_gating)
        frame_depth_map = self.memm(x)
        frame_attention_map = self.attention_gating_network(x)
        depth_map = (frame_depth_map * frame_attention_map).sum(dim=1)
        pred = self.classification_head(depth_map)

        return pred, depth_map


if __name__ == "__main__":
    model = ATRFAS(num_class=3, num_frames=1, return_proba=False, infer_type='training')

    images = []
    image = cv2.imread('../../images/frame_1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    basic_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Augmentation transformations with low intensity
    augmentations = transforms.Compose([
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3))  # Light blur
        ], p=0.3),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=(0.8, 1.3))  # Subtle brightness adjustment
        ], p=0.4),
        transforms.RandomApply([
            transforms.ColorJitter(hue=(-0.05, 0))  # Subtle hue adjustment
        ], p=0.4),
        transforms.RandomApply([
            transforms.ColorJitter(saturation=(0.8, 1.3))  # Subtle saturation adjustment
        ], p=0.4),
        transforms.RandomApply([
            transforms.RandomPosterize(bits=7, p=0.3)  # Light posterization
        ], p=0.3),
        transforms.RandomApply([
            transforms.RandomAdjustSharpness(2, p=1)  # Light posterization
        ], p=0.3),
    ])

    image = augmentations(image)
    image = basic_transform(image)
    images.append(image)
    images = torch.stack(images).float()
    x = images.unsqueeze(0)  # [B, N, C, H, W]

    x = torch.randn(2, 1, 3, 256, 256)

    model.train()
    pred, frame_depth_map = model(x)

    model.eval()
    dummy_input = torch.randn(1, 1, 3, 256, 256)
    torch.onnx.export(model, dummy_input, "out.onnx", keep_initializers_as_inputs=False, verbose=False,
                      opset_version=12)

    import onnx

    onnx_model = onnx.load("out.onnx")
    # from onnxsim import simplify
    # onnx_model, check = simplify(onnx_model)
    # assert check, "Simplified ONNX model could not be validated"
    # import onnxoptimizer
    # onnx_model = onnxoptimizer.optimize(onnx_model)
    # onnx.save(onnx_model, "out.onnx")

    from onnx import numpy_helper

    total_parameters = 0
    for initializer in onnx_model.graph.initializer:
        total_parameters += numpy_helper.to_array(initializer).size
    final_cls = model(x)
    final_cls = final_cls
