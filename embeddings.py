from urllib.request import urlopen
from PIL import Image
import glob
import timm
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

paths = sorted(glob.glob("data/*"))
images = [Image.open(path) for path in paths]
images = [img.convert("RGB") for img in images]


img = Image.open(
    urlopen(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    )
)


model = timm.create_model(
    "maxvit_tiny_tf_224.in1k",
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

outputs = [
    model(transforms(img).unsqueeze(0)) for img in images
]  # output is (batch_size, num_features) shaped tensor

# or equivalently (without needing to set num_classes=0)

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

cos(output[0], output[0]).detach().numpy()

overview = pd.DataFrame(similarity)
names = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
overview.columns = names
overview.index = names

overview

plt.imshow(images[0])
plt.show()

np.array(images[0]).shape
