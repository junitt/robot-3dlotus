import clip
import torch

def get_clip_model(device="cuda"):
    try:
        clip_model, _ = clip.load("RN50", device="cpu")  # CLIP-ResNet50
        clip_model = clip_model.to(device)
        clip_model.eval()
    except RuntimeError:
        print("WARNING: Setting Clip to None. Will not work if replay not on disk.")
        clip_model = None
    return clip_model

#consider reconstruct
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, emb

def _tokenizer_tensor(description,device="cuda"):
    tokens = clip.tokenize([description]).numpy()
    token_tensor = torch.from_numpy(tokens).to(device)
    return token_tensor

def get_embed(clip_model,description,device="cuda"):
    token_tensor = _tokenizer_tensor(description,device)
    with torch.no_grad():
        lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
    return lang_embs[0].float().detach().cpu().numpy()