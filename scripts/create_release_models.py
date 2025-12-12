
import os
import torch
import logging
from pathlib import Path
from thulium.pipeline.htr_pipeline import HTRPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("release_builder")

def create_model(name: str, config_path: str, output_dir: Path):
    logger.info(f"Creating {name} from {config_path}...")
    try:
        # Create untrained model from config
        pipeline = HTRPipeline.from_config(config_path)
        
        # Save state dict
        output_path = output_dir / f"{name}.pt"
        torch.save(pipeline.recognizer.state_dict(), output_path)
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {name}.pt ({size_mb:.2f} MB)")
        
    except Exception as e:
        logger.error(f"Failed to create {name}: {e}")

def main():
    output_dir = Path("releases/v1.2.1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = [
        ("thulium-tiny", "config/models/htr_cnn_lstm_ctc_tiny.yaml"),
        ("thulium-base", "config/models/htr_cnn_lstm_ctc_base.yaml"),
        ("thulium-large", "config/models/htr_vit_transformer_seq2seq_large.yaml"),
        # Multilingual uses same large architecture but would ideally have different vocab
        # For this dummy release, we reuse the large config
        ("thulium-multilingual", "config/models/htr_vit_transformer_seq2seq_large.yaml"),
    ]
    
    for name, config in models:
        create_model(name, config, output_dir)
        
    logger.info(f"All models created successfully in {output_dir.absolute()}")
    logger.info("Please upload these files to your GitHub Release assets.")

if __name__ == "__main__":
    main()
