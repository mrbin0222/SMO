import clip
import torch

# Load language-vision model, e.g. CLIP, CLIP-Surgery, ALBEF, BLIP, BLIP2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("../checkpoints/ViT-B-32.pt", device=device)
print(type(model.eval()))

@torch.no_grad()
def retriev(elements: list, search_text: str) -> int:
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)

