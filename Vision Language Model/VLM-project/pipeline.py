import torch
import json
import yaml
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, AutoTokenizer
from typing import Optional, List, Dict, Union
import logging
from vlm_model import VisionLanguageModel
from train import SyntheticVLDataset, VLMTrainer, VisionLanguageDataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class VLMInference:
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        lm_model_name: str = "HuggingFaceTB/SmolLM2-360M",
        vit_model_name: str = "google/vit-base-patch16-224",
        blip_model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = "auto",
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        logger.info(f"Loading model on {self.device}...")
        self.model = VisionLanguageModel(
            lm_model_name=lm_model_name,
            vit_model_name=vit_model_name,
            blip_model_name=blip_model_name,
        )

        if checkpoint_dir is not None:
            self._load_checkpoint(checkpoint_dir)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = BlipProcessor.from_pretrained(blip_model_name)
        logger.info("Model ready.")

    @classmethod
    def from_config(cls, config: dict, checkpoint_dir: Optional[str] = None):
        m = config["model"]
        inf = config.get("inference", {})
        return cls(
            checkpoint_dir=checkpoint_dir,
            lm_model_name=m["lm_model_name"],
            vit_model_name=m["vit_model_name"],
            blip_model_name=m["blip_model_name"],
            max_new_tokens=inf.get("max_new_tokens", 80),
            temperature=inf.get("temperature", 0.7),
            do_sample=inf.get("do_sample", True),
        )

    def _load_checkpoint(self, checkpoint_dir: str):
        ckpt = Path(checkpoint_dir)
        self.model.vision_encoder.load_state_dict(torch.load(ckpt / "vision_encoder.pt", map_location="cpu"))
        self.model.vision_query = torch.load(ckpt / "vision_query.pt", map_location="cpu")
        self.model.cross_attention.load_state_dict(torch.load(ckpt / "cross_attention.pt", map_location="cpu"))
        self.model.vision_token_compressor.load_state_dict(torch.load(ckpt / "compressor.pt", map_location="cpu"))
        lm_path = ckpt / "lm"
        if lm_path.exists():
            from transformers import AutoModelForCausalLM
            self.model.lm = AutoModelForCausalLM.from_pretrained(lm_path)
        tokenizer_path = ckpt / "tokenizer"
        if tokenizer_path.exists():
            self.model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Loaded checkpoint: {checkpoint_dir}")

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs.pixel_values.to(self.device)

    @torch.no_grad()
    def answer_question(self, image: Union[str, Image.Image], question: str, **gen_kwargs) -> str:
        pixel_values = self.preprocess_image(image)
        prompt = f"<image> {question} "
        input_ids = self.model.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated_ids = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            max_new_tokens=gen_kwargs.get("max_new_tokens", self.max_new_tokens),
            temperature=gen_kwargs.get("temperature", self.temperature),
            do_sample=gen_kwargs.get("do_sample", self.do_sample),
        )
        output_text = self.model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if question in output_text:
            output_text = output_text.split(question)[-1].strip()
        return output_text

    @torch.no_grad()
    def caption_image(self, image: Union[str, Image.Image]) -> str:
        return self.answer_question(image, "Describe this image in detail:", do_sample=False)

    def get_vision_embedding(self, image: Union[str, Image.Image]) -> torch.Tensor:
        pixel_values = self.preprocess_image(image)
        with torch.no_grad():
            return self.model.encode_vision(pixel_values)


def run_stage(
    model: VisionLanguageModel,
    processor,
    stage_cfg: dict,
    data_cfg: dict,
    ckpt_cfg: dict,
    stage_name: str,
):
    logger.info(f"=== Starting {stage_name} ===")

    model.lm.requires_grad_(not stage_cfg["freeze_lm"])
    for param in model.vision_encoder.parameters():
        param.requires_grad = True
    for param in model.cross_attention.parameters():
        param.requires_grad = True
    model.vision_query.requires_grad_(True)

    n = stage_cfg["num_synthetic_samples"]
    val_n = max(1, int(n * data_cfg.get("val_split", 0.1)))

    if data_cfg.get("use_synthetic", True) or data_cfg.get("data_path") is None:
        train_dataset = SyntheticVLDataset(processor, model.tokenizer, num_samples=n)
        val_dataset = SyntheticVLDataset(processor, model.tokenizer, num_samples=val_n)
    else:
        train_dataset = VisionLanguageDataset(
            data_cfg["data_path"], processor, model.tokenizer,
            max_text_length=data_cfg.get("max_text_length", 128), split="train"
        )
        val_dataset = VisionLanguageDataset(
            data_cfg["data_path"], processor, model.tokenizer,
            max_text_length=data_cfg.get("max_text_length", 128), split="val"
        )

    output_dir = str(Path(ckpt_cfg["output_dir"]) / stage_name)

    trainer = VLMTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        batch_size=stage_cfg["batch_size"],
        num_epochs=stage_cfg["num_epochs"],
        learning_rate=stage_cfg["learning_rate"],
        vision_lr_multiplier=stage_cfg.get("vision_lr_multiplier", 0.1),
        warmup_steps=stage_cfg.get("warmup_steps", 100),
        gradient_accumulation_steps=stage_cfg["gradient_accumulation_steps"],
        max_grad_norm=stage_cfg.get("max_grad_norm", 1.0),
        save_every_n_steps=ckpt_cfg.get("save_every_n_steps", 200),
        eval_every_n_steps=ckpt_cfg.get("eval_every_n_steps", 100),
    )
    trainer.train()
    return trainer, output_dir


def run_training_pipeline(config: dict):
    m_cfg = config["model"]
    t_cfg = config["training"]
    d_cfg = config["data"]
    ckpt_cfg = config["checkpoint"]

    logger.info("Initializing model...")
    model = VisionLanguageModel(
        lm_model_name=m_cfg["lm_model_name"],
        vit_model_name=m_cfg["vit_model_name"],
        blip_model_name=m_cfg["blip_model_name"],
        fusion_type=m_cfg.get("fusion_type", "concat"),
        freeze_lm=True,
        num_vision_tokens=m_cfg.get("num_vision_tokens", 64),
    )

    processor = BlipProcessor.from_pretrained(m_cfg["blip_model_name"])

    trainer1, stage1_dir = run_stage(model, processor, t_cfg["stage1"], d_cfg, ckpt_cfg, "stage1")
    trainer2, stage2_dir = run_stage(model, processor, t_cfg["stage2"], d_cfg, ckpt_cfg, "stage2")

    logger.info(f"Training complete. Final checkpoint: {stage2_dir}")
    return stage2_dir


def run_inference_demo(config: dict, checkpoint_dir: str):
    from synthetic_image_gen import generate_sample
    inferencer = VLMInference.from_config(config, checkpoint_dir=checkpoint_dir)

    for idx in [0, 3, 5, 7]:
        img, true_caption, question = generate_sample(idx)
        answer = inferencer.answer_question(img, question)
        logger.info(f"Q : {question}")
        logger.info(f"A : {answer}")
        logger.info(f"GT: {true_caption}\n")

    embedding = inferencer.get_vision_embedding(generate_sample(0)[0])
    logger.info(f"Vision embedding shape: {embedding.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", choices=["train", "infer", "both"], default="both")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode in ("train", "both"):
        final_ckpt = run_training_pipeline(cfg)

    ckpt = args.checkpoint_dir or (final_ckpt if args.mode == "both" else None)
    if args.mode in ("infer", "both"):
        run_inference_demo(cfg, checkpoint_dir=ckpt)
