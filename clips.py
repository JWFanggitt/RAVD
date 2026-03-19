import torch
import clip as clip_model
device=torch.device("cuda",0)
texts=["What is the cause of this accident"]
image=torch.randn(1,3,224,224).to(device)
model,preprocess=clip_model.load("ViT-L/14")
model=model.to(device)
with torch.no_grad():
    text_inputs=clip_model.tokenize(texts).to(device)
    text_feastures=model.encode_text( text_inputs)
    image_features=model.encode_image(image)
    print(text_feastures.shape)
    print(image_features.shape)