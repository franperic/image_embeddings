from PIL import Image
import glob
import os
import timm
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd


def load_images(path):
    paths = sorted(glob.glob(path))
    images = [Image.open(img).convert("RGB") for img in paths]
    return images


def generate_outputs(model, images, transforms):
    # Initialize an empty list to store the outputs
    outputs = []

    # Iterate over the images and compute the combined output
    for img in images:
        tensor = transforms(img).unsqueeze(0)
        output_features = model.forward_features(tensor)
        output = model.forward_head(output_features, pre_logits=True)
        outputs.append(output)

    return outputs


def calculate_similarity(outputs):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = {}
    for i, embedding in enumerate(outputs):
        similarity[i] = [cos(embedding, emb).item() for emb in outputs]
    return similarity


def highlight_max(s):
    is_max = s == s.sort_values(ascending=False).iloc[1]
    return ["background-color: yellow" if v else "" for v in is_max]


def main():
    PATH = os.path.join("data", "random", "*")
    images = load_images(PATH)

    model = timm.create_model(
        "maxvit_tiny_tf_224.in1k",
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    )
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    outputs = generate_outputs(model, images, transforms)
    similarity = calculate_similarity(outputs)

    overview = pd.DataFrame(similarity)
    names = ["Cat", "House", "Lion", "Lions"]
    overview.columns = names
    overview.index = names

    # apply the highlight function to the dataframe
    styled_df = overview.style.apply(highlight_max)
    return styled_df


if __name__ == "__main__":
    main()
