import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms

from src.dataloader import VireoLoader
from src.model_clip import Recognition

@st.cache_data
def get_ingre_probability(ingres, uploaded_image):
    # Load CLIP model
    device = "cpu"
    model, preprocess = ja_clip.load("rinna/japanese-cloob-vit-b-16", device=device)
    tokenizer = ja_clip.load_tokenizer()

    input_image = preprocess(uploaded_image).unsqueeze(0).to(device)
    search = model.get_image_features(input_image).cpu()

    encodings = ja_clip.tokenize(
        texts=[f"{ing}を使った料理" for ing in ingres],
        max_seq_len=77,
        device=device,
        tokenizer=tokenizer,  # this is optional. if you don't pass, load tokenizer each time
    )

    with torch.no_grad():
        ingre_text_features = model.get_text_features(**encodings)
        image_features = search.to(device)
        # text_probs = (100.0 * image_features @ ingre_text_features.T).softmax(dim=-1)
        text_probs = 100.0 * image_features @ ingre_text_features.T
        # values, indices = text_probs.topk(10)
        # for item in indices[0]:
        #     res_label.append(item.cpu().numpy().astype(int))

    return text_probs


@st.cache_data
def get_ingre_prob_from_model(uploaded_image):
    model_path = "Models/clip_v32_epoch21_20.938.pth"

    device = torch.device("cpu")
    model = Recognition()
    model = nn.DataParallel(model)
    model.to(device)
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))
    # model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    criterions = [
        nn.CrossEntropyLoss().to("cpu"),
        nn.BCELoss(reduction="none").to("cpu"),
    ]

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    test_loader = torch.utils.data.DataLoader(
        VireoLoader(
            uploaded_image,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=1,
        shuffle=True,
        timeout=1000,
        num_workers=1,
        # pin_memory=True,
    )

    for i, inputs in enumerate(test_loader):
        imgs = inputs[0].to("cpu")
        outputs = model(imgs)

    text_probs = outputs[1].cpu()
    # text_probs = torch.from_numpy(np.load('/home/l_wang/vireofood251/RA-CLIP/ingre_feature_pasta.npy'))
    min_value = torch.min(text_probs)
    abs_min_value = torch.abs(min_value)
    normalized_tensor = text_probs + abs_min_value
    # to positive

    min_normalized_value = torch.min(normalized_tensor)
    max_normalized_value = torch.max(normalized_tensor)
    normalized_tensor = (normalized_tensor - min_normalized_value) / (
        max_normalized_value - min_normalized_value
    )
    # 0-1

    return normalized_tensor

@st.cache_data
def get_current_candidate(candidate_nums, uploaded_image, mask):
    text_probs = get_ingre_prob_from_model(uploaded_image)
    probability_scores = [item for sublist in text_probs.tolist() for item in sublist]
    pos_probability = get_pos_probability(probability_scores)
    cur_prob = pos_probability * mask
    top_k_indices = np.argsort(cur_prob)[-candidate_nums:][::-1]
    return top_k_indices.tolist()

@st.cache_data
def get_pos_probability(text_probs):
    alpha = 0.1
    flipped_relative_pos = relative_pos()
    text_probs = np.array(text_probs)
    flipped_relative_pos = np.array(flipped_relative_pos)
    final_probs = (1 - alpha) * text_probs + alpha * flipped_relative_pos
    # st.write("text_probs:",text_probs)
    # st.write("flipped_relative_pos",flipped_relative_pos)
    return final_probs.tolist()

@st.cache_data
def relative_pos():
    relative_pos = np.load("Labels/relative_pos.npy", allow_pickle=True)
    flipped_relative_pos = 1 - relative_pos
    return flipped_relative_pos

