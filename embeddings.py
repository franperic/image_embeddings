from PIL import Image
import glob
import timm
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd

# Example Images
paths = sorted(glob.glob("data/image*"))
images = [Image.open(path) for path in paths]
images = [img.convert("RGB") for img in images]

names = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(images[i])
    ax.title.set_text(f"Image {names[i]}")
    ax.axis("off")


model = timm.create_model(
    "maxvit_tiny_tf_224.in1k",
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

outputs = [model.forward_features(transforms(img).unsqueeze(0)) for img in images]
# output is unpooled, a (1, 1025, 768) shaped tensor

output = [model.forward_head(output, pre_logits=True) for output in outputs]
# output is a (1, num_features) shaped tensor

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
similarity = {}
for i in range(len(output)):
    similarity[i] = [
        cos(output[i], embedding).detach().numpy()[0] for embedding in output
    ]

overview = pd.DataFrame(similarity)
names = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
overview.columns = names
overview.index = names


# create a function to highlight the biggest number in each column
def highlight_max(s):
    is_max = s == s.sort_values(ascending=False).iloc[1]
    return ["background-color: yellow" if v else "" for v in is_max]


# apply the highlight function to the dataframe
styled_df = overview.style.apply(highlight_max)
styled_df


# Random Images
paths = sorted(glob.glob("data/random/*"))
images = [Image.open(img) for img in paths]
images = [img.convert("RGB") for img in images]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(images[i])
    ax.axis("off")

outputs = [model.forward_features(transforms(img).unsqueeze(0)) for img in images]
# output is unpooled, a (1, 1025, 768) shaped tensor

output = [model.forward_head(output, pre_logits=True) for output in outputs]
# output is a (1, num_features) shaped tensor


cos = nn.CosineSimilarity(dim=1, eps=1e-6)
similarity = {}
for i in range(len(output)):
    similarity[i] = [
        cos(output[i], embedding).detach().numpy()[0] for embedding in output
    ]

overview = pd.DataFrame(similarity)
names = ["Cat", "House", "Lion", "Lions"]
overview.columns = names
overview.index = names


# create a function to highlight the biggest number in each column
def highlight_max(s):
    is_max = s == s.sort_values(ascending=False).iloc[1]
    return ["background-color: yellow" if v else "" for v in is_max]


# apply the highlight function to the dataframe
styled_df = overview.style.apply(highlight_max)
styled_df
