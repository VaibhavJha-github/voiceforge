"""
VoiceForge — Multi-character batch TTS powered by Fish-Speech.

Usage:
    python tools/voiceforge_app.py [--half] [--compile] [--device cuda]

This loads the Fish-Speech model and launches the VoiceForge UI.
Clone character voices, write multi-character scripts, batch generate audio.
"""

import os
from argparse import ArgumentParser
from pathlib import Path

import pyrootutils
import torch
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from tools.voiceforge.ui import build_voiceforge_app

# Make einx happy
os.environ["EINX_FILTER_TRACEBACK"] = "false"


def parse_args():
    parser = ArgumentParser(description="VoiceForge — Multi-character batch TTS")
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/s2-pro",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/s2-pro/codec.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true", help="Use fp16 (saves VRAM)")
    parser.add_argument("--compile", action="store_true", help="torch.compile optimization")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create Gradio share link")
    return parser.parse_args()


def main():
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    # Auto-detect device
    if args.device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            args.device = "mps"
            logger.info("CUDA not available, using MPS (Apple Silicon).")
        else:
            args.device = "cpu"
            logger.warning("No GPU available, using CPU. This will be SLOW.")

    if args.device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f}GB VRAM)")
        if vram_gb < 16:
            logger.info("Less than 16GB VRAM detected — enabling --half automatically.")
            args.precision = torch.half

    logger.info("Loading Fish-Speech LLaMA model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        compile=args.compile,
    )

    logger.info("Loading VQ-GAN decoder model...")
    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )

    logger.info("Models loaded. Warming up...")

    # Create inference engine
    inference_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=args.compile,
        precision=args.precision,
    )

    # Warm up with a dry run
    list(
        inference_engine.inference(
            ServeTTSRequest(
                text="Hello world.",
                references=[],
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.5,
                temperature=0.7,
                format="wav",
            )
        )
    )

    logger.info("Warm-up complete. Launching VoiceForge UI...")

    # Get sample rate from decoder
    sample_rate = decoder_model.sample_rate

    # Build and launch
    app = build_voiceforge_app(
        inference_engine=inference_engine,
        sample_rate=sample_rate,
    )

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
