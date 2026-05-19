"""
Model definitions for the project.
"""
from utils.imports import *

class SmoothBCEWithLogitsLoss(nn.Module):
    """
    Binary cross-entropy loss with logits and manual label smoothing.
    """
    def __init__(self, pos_weight=None, smoothing=0.1):
        super().__init__()
        self.pos_weight = pos_weight
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # Smooth targets: positive labels -> 1 - smoothing/2, negatives -> smoothing/2
        smoothed = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, smoothed, pos_weight=self.pos_weight)


class CLIPForBinary(nn.Module):
    def __init__(self, dropout_rate=0.5, pretrained=False, num_classes=1):
        super().__init__()
        if pretrained:
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            config = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
            self.clip = CLIPModel(config)
        
        # Replace classifier to handle multiclass if needed
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.clip.config.projection_dim, num_classes)
        )

    def forward(self, pixel_values):
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_embeds = self.clip.visual_projection(vision_outputs.pooler_output)
        return self.classifier(image_embeds)


# Smooth BCE Loss with label smoothing for binary tasks
