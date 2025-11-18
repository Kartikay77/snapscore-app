import os
import warnings

import torch
from PIL import Image
import streamlit as st

from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ------------------------ NLTK SETUP ------------------------
for pkg in ["punkt", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

# ------------------------ DEVICE ------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ------------------------ CACHED MODELS ------------------------
@st.cache_resource
def load_sd_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(DEVICE)
    return pipe

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    return processor, blip

# ------------------------ CAPTIONING ------------------------
def blip_caption(image: Image.Image) -> str:
    processor, blip = load_blip()
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    out = blip.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)

# ------------------------ METRICS (SAME AS YOUR SCRIPT) ------------------------
def bleu_score(ref: str, cand: str) -> float:
    ref_tokens = [nltk.word_tokenize(ref.lower())]
    cand_tokens = nltk.word_tokenize(cand.lower())
    return sentence_bleu(
        ref_tokens, cand_tokens,
        smoothing_function=SmoothingFunction().method4
    )

def rouge_f1_avg(ref: str, cand: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, cand)
    return (
        scores["rouge1"].fmeasure
        + scores["rouge2"].fmeasure
        + scores["rougeL"].fmeasure
    ) / 3.0

def meteor(ref: str, cand: str) -> float:
    return meteor_score([ref.lower().split()], cand.lower().split())

def cosine_sim(ref: str, cand: str) -> float:
    vec = TfidfVectorizer().fit_transform([ref.lower(), cand.lower()])
    return cosine_similarity(vec[0:1], vec[1:2])[0][0]

def snapscore(ref: str, cand: str) -> dict:
    scores = {
        "BLEU":   bleu_score(ref, cand),
        "ROUGE":  rouge_f1_avg(ref, cand),
        "METEOR": meteor(ref, cand),
        "cosine": cosine_sim(ref, cand),
    }
    weights = {"BLEU": 0.20, "ROUGE": 0.20, "METEOR": 0.25, "cosine": 0.35}
    snap = sum(scores[k] * weights[k] for k in weights)
    return {"scores": scores, "SNAP": snap}

# ------------------------ STREAMLIT APP ------------------------
st.set_page_config(page_title="SnapScore Image App", layout="centered")

st.title("SnapScore Image Generator & Evaluator")

st.markdown(
    "Enter a prompt, generate an image with Stable Diffusion, "
    "auto-caption it with BLIP, and compute BLEU, ROUGE, METEOR, cosine similarity, and a combined **SnapScore**."
)

prompt = st.text_area(
    "Prompt used for image generation:",
    "A stylish young man sitting inside a car in New York City, looking out the window. "
    "Ultra realistic, 4K photograph, cinematic lighting, Times Square lights reflecting on the car windows, "
    "shallow depth of field, detailed textures, natural skin tones, premium DSLR look."
)

col_settings = st.columns(2)
with col_settings[0]:
    steps = st.slider("Diffusion steps", 10, 50, 25)
with col_settings[1]:
    guidance_scale = st.slider("Guidance scale", 1.0, 15.0, 7.5)

if st.button("Generate Image and Compute SnapScore"):
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        # ---- Generate image ----
        with st.spinner("Generating image with Stable Diffusion..."):
            pipe = load_sd_pipeline()
            image = pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale
            ).images[0]

        st.subheader("Prompt Used")
        st.write(prompt)

        st.subheader("️Generated Image")
        st.image(image, caption="Generated from the prompt above", use_column_width=True)

        # ---- Caption + scores ----
        with st.spinner("Captioning image with BLIP and computing scores..."):
            caption = blip_caption(image)
            result = snapscore(prompt, caption)

        st.subheader("BLIP Caption")
        st.write(caption)

        st.subheader("Metric Scores")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("BLEU",   f"{result['scores']['BLEU']:.4f}")
            st.metric("ROUGE",  f"{result['scores']['ROUGE']:.4f}")
        with col2:
            st.metric("METEOR", f"{result['scores']['METEOR']:.4f}")
            st.metric("Cosine", f"{result['scores']['cosine']:.4f}")

        st.subheader("SnapScore (Weighted)")
        st.metric("SnapScore", f"{result['SNAP']:.4f}")
