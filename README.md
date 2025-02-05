# wd-tagger-rs

An inference tool of [WaifuDiffusion Tagger](https://huggingface.co/spaces/SmilingWolf/wd-tagger).

> [!IMPORTANT]
> WIP. 

## Usage (Experimental)

Get Rust toolchain:

See https://www.rust-lang.org/tools/install

### With CPU (recommended)

To build:

```bash
cargo install --git https://github.com/p1atdev/wd-tagger-rs
```

To run:

```bash
tagger v3 ./assets/sample1_3x1024x1024.webp
```

Output:

```
[src/main.rs:183:13] result = TaggingResult {
    rating: {
        "general": 0.91256857,
    },
    character: {},
    general: {
        "1girl": 0.996445,
        "solo": 0.977317,
        "double_bun": 0.94901526,
        "hair_bun": 0.94456,
        "twintails": 0.9389738,
        "pink_hair": 0.93058735,
        "fang": 0.8859673,
        "smile": 0.88062656,
        "pink_eyes": 0.8463925,
        "looking_at_viewer": 0.83266306,
...
```


## Models

### v3 family

You can use v3 family models with the `tagger v3` command, and you can specify the model with the `--model` option.

- `--model`
  - `vit`: SmilingWolf/wd-vit-tagger-v3
  - `swin-v2`: SmilingWolf/wd-swin-v2-tagger-v3 (default)
  - `convnext`: SmilingWolf/wd-convnext-tagger-v3
  - `vit-large`: SmilingWolf/wd-vit-large-tagger-v3
  - `eva02-large`: SmilingWolf/wd-eva02-large-tagger-v3

Example:
```bash
tagger v3 ./assets/sample1_3x1024x1024.webp --model eva02-large
```

See `tagger v3 --help` for more details.

### Run custom models

You can use the custom models with `tagger custom` command, that is on HuggingFace and the same format of the original model.

- Example: [deepghs/idolsankaku-eva02-large-tagger-v1](https://huggingface.co/deepghs/idolsankaku-eva02-large-tagger-v1)

```bash
tagger custom ./assets/sample1_3x1024x1024.webp \
  --repo-id deepghs/idolsankaku-eva02-large-tagger-v1 
```

```bash
Target device: <CPU>
[src/cli/main.rs:112:13] &result = TaggingResult {
    rating: {
        "safe": 0.94494337,
    },
    character: {},
    general: {
        "twintails": 0.95630574,
        "pink_hair": 0.91894686,
        "female": 0.8313366,
        "solo": 0.8135544,
        "1girl": 0.74666,
        "looking_at_viewer": 0.6675732,
        "ribbon": 0.6159363,
        "asian": 0.52826667,
        "female_only": 0.5272801,
        "double_bun": 0.46635512,
        "long_hair": 0.42993295,
        "blouse": 0.41456583,
        "east_asian": 0.37745702,
        "japanese": 0.35556853,
    },
}
```

See `tagger custom --help` for more details.

## Save the prediction result

### as JSON

If you specified `--output` option, tagger will save the result as JSON in default.

```bash
tagger v3 ./assets/sample1_3x1024x1024.webp \
  --output ./output.json
```

Or you can specify the output format explicitly:

```bash
tagger v3 ./assets/sample1_3x1024x1024.webp \
  --output ./output.json \
  --format json
```

The json file includes all of the prediction results. 


### as Caption

If you specify `--output` option, tagger will save the result as JSON in default.

```bash
tagger v3 ./assets/sample1_3x1024x1024.webp \
  --output ./output.json
```

Or you can specify the output format explicitly:

```bash
tagger v3 ./assets/sample1_3x1024x1024.webp \
  --output ./output.json \
  --format json
```

If you don't specify the `--output` option, tagger will save to the same directory of the input file.

```bash
tagger v3 ./assets/sample1_3x1024x1024.webp \
  --format json
```
Tagger saves to `./assets/sample1_3x1024x1024.json`.



## Experimental execution devices

### With CUDA 

#### Prerequisites

##### cuDNN

cuDDN 9.x **MUST** be installed. You can get it from here:

https://developer.nvidia.com/cudnn-downloads

##### onnxruntime 

Downlaod prebuilt onnxruntime from ONNX Runtime's releases. (e.g. `onnxruntime-linux-x64-gpu-1.19.0.tgz`):

https://github.com/microsoft/onnxruntime/releases/tag/v1.19.0 

Then extract it and place files to `~/.local/share`, and set `LD_LIBRARY_PATH`.

For example:

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.19.0/onnxruntime-linux-x64-gpu-1.19.0.tgz
tar -xvf onnxruntime-linux-x64-gpu-1.19.0.tgz
mkdir -p ~/.local/share/wdtagger/onnxruntime
mv onnxruntime-linux-x64-gpu-1.19.0 ~/.local/share/wdtagger/onnxruntime/1.19.0
rm onnxruntime-linux-x64-gpu-1.19.0.tgz
```

Add the following to your `.bashrc` or `.zshrc`:

```bash
# wdtagger
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/share/wdtagger/onnxruntime/1.19.0/lib
```

> [!NOTE]
> Please check that you are specifying the `lib` directory, not the root directory of the extracted onnxruntime.

To apply:

```bash
source ~/.bashrc
```

#### Build

To build:

```bash
cargo install --path . --features cuda
```

To run:

```bash
tagger v3 ./assets/sample1_3x1024x1024.webp \
    --devices 0 \
    --model vit-large # vit, swin-v2, convnext, vit-large, eva02-large
```

#### Docker

This is just PoC.

Using docker:

```yml
services:
  cuda:
    build:
      context: .
      dockerfile: ./docker/Dockerfile.cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - type: bind
        source: ./assets
        target: /workspace/assets
      - type: bind # to use huggingface cache
        source: ~/.cache/huggingface
        target: /root/.cache/huggingface

    command: ["./tagger"] 
```

To run:

```bash
docker compose run cuda ./tagger ./assets/sample1_3x1024x1024.webp 
```

To down:
```bash
docker compose down --remove-orphans
```

### With TensorRT 

#### Prerequisites

##### TensorRT

You need at least `libnvinfer`. You can get it from here:

https://developer.nvidia.com/tensorrt/download/10x

#### Build

```bash
cargo install --path . --features tensorrt
```

```bash
tagger v3 ./assets/sample1_3x1024x1024.webp \
    --devices 0 \
    --model eva02-large
```

> [!NOTE]
> Currently TensorRT mode is not so fast as CUDA mode.

### With CoreML

#### Build

```bash
cargo install --path . --features coreml
```

```bash
tagger v3 ./assets/sample1_3x1024x1024.webp \
    --model eva02-large
```

