<div align="center">

# 🌐 OverWorld Inference Engine

  Inference Engine for World Models
</div>


## Usage

#### Setup
```
# Install
pip install "world_engine@git+https://github.com/Wayfarer-Labs/world_engine.git"

# Specify HuggingFace Token (https://huggingface.co/settings/tokens)
export HF_TOKEN=<your access token>
```

#### Run
```
from world_engine import WorldEngine, CtrlInput

# Create inference engine
engine = WorldEngine("OpenWorldLabs/CoD-Img-Base", device="cuda")

# Specify a prompt
engine.set_prompt("A fun game")

# Generate 3 video frames conditioned on controller inputs
for controller_input in [
		CtrlInput(button={48, 42}, mouse=[0.4, 0.3]),
		CtrlInput(mouse=[0.1, 0.2]),
		CtrlInput(button={95, 32, 105}),
]:
	img = engine.gen_frame(ctrl=controller_input)
```

## Scope

### What this package is
A library that takes care of model and inference plumbing
- Manages the whole frame-generation pipeline, including the DiT, autoencoder, text encoder, and KV cache
- Lets you specify in a model via its URI and pass in controller inputs, text prompts, and images directly

In scope:
- Optimized implementations for Nvidia, AMD, Apple Silicon, etc
- Support for both consumer and data center GPUs
- Loading base World Models and LoRA adapters
- Generating frame images conditioned on prior frames, control inputs, and text prompts

### What this package isn't
This isn't a fully featured client, it's a core library.

Out of scope:
- Rendering or displaying video and images
- Reading controller, keyboard, or mouse input from devices
- FAL, and other integrations

Anything out of scope can be added to `examples/`, which isn't part of the `world_engine.*` package.
