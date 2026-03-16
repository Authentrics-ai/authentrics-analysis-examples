import torch
import torchvision

classes = [
    "A10",
    "AH64",
    "B1",
    "B52",
    "C130",
    "C17",
    "C2",
    "EF2000",
    "F15",
    "F16",
    "F18",
    "F22",
    "F35",
    "F4",
    "J10",
    "J20",
    "JAS39",
    "Rafale",
    "US2",
    "V22",
]


class MilAirModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.num_classes = len(classes)
        self.squeeze_edit_model = torch.hub.load(
            "pytorch/vision:v0.20.1",
            "efficientnet_b3",
            weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT,
        )

        self.squeeze_edit_model.classifier[1] = torch.nn.Linear(1536, 256)
        self.squeeze_edit_model.classifier.extend(
            [
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.45),
                torch.nn.Linear(256, self.num_classes),
            ]
        )

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.squeeze_edit_model(x)
