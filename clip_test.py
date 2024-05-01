from PIL import Image
import torch
import japanese_clip as ja_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
# ja_clip.available_models()
# ['rinna/japanese-clip-vit-b-16', 'rinna/japanese-cloob-vit-b-16']
# If you want v0.1.0 models, set `revision='v0.1.0'`
model, preprocess = ja_clip.load("rinna/japanese-clip-vit-b-16", cache_dir="/tmp/japanese_clip", device=device)
tokenizer = ja_clip.load_tokenizer()

image = preprocess(Image.open("recipe1-1.jpg")).unsqueeze(0).to(device)
encodings = ja_clip.tokenize(
    texts=["ロールキャベツ", "ミートソーススパゲティ", "カレーライス"],
    max_seq_len=77,
    device=device,
    tokenizer=tokenizer, # this is optional. if you don't pass, load tokenizer each time
)

with torch.no_grad():
    image_features = model.get_image_features(image)
    text_features = model.get_text_features(**encodings)
    
    softmax_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    logit_probs = (image_features @ text_features.T).softmax(dim=0)

print("Label probs:", softmax_probs)  # prints: [[1.0, 0.0, 0.0]]
print("Label probs:", logit_probs)  # prints: [[1.0, 0.0, 0.0]]
