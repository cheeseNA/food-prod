import torch
import torch.nn as nn

# import clip
from transformers import CLIPModel, CLIPProcessor


# from args import get_parser
# parser = get_parser()
# opts = parser.parse_args()


class Recognition(nn.Module):
    def __init__(self):
        super(Recognition, self).__init__()

        # self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

        # 将视觉编码器的输出维度适应您的任务（根据您的 opts.num_cat 和 opts.num_ing）
        self.visual_embedding_dim = 512  # 根据 CLIP 模型的配置确定维度
        self.cat_pred = nn.Linear(self.visual_embedding_dim, 82)
        self.ing_pred = nn.Linear(self.visual_embedding_dim, 588)
        self.max = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, input):
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(input)

        cat_pred = self.cat_pred(image_features)
        cat_pred = self.max(cat_pred.view(input.size(0), 82, 1, 1)).view(
            input.size(0), -1
        )
        ing_pred = self.ing_pred(image_features)
        ing_pred = self.max(ing_pred.view(input.size(0), 588, 1, 1)).view(
            input.size(0), -1
        )
        return [cat_pred, ing_pred]
