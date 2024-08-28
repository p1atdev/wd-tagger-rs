# wd-tagger-rs

An inference tool of [WaifuDiffusion Tagger](https://huggingface.co/spaces/SmilingWolf/wd-tagger).

> [!IMPORTANT]
> WIP. Not ready for use.

## Usage (Experimental)

Get Rust toolchain:

See https://www.rust-lang.org/tools/install

### With CPU

(Only tested on Ubuntu 24.04)

To build:

```bash
cargo install --git https://github.com/p1atdev/wd-tagger-rs
```

To run:

```bash
tagger ./assets/sample1_3x1024x1024.webp
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

### With CUDA

Very experimental.

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
tagger ./assets/sample1_3x1024x1024.webp \
    --devices 0 \
    --v3 vit-large # vit, swin-v2, convnext, vit-large, eva02-large
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

Very experimental.

#### Prerequisites

##### TensorRT

You need at least `libnvinfer`. You can get it from here:

https://developer.nvidia.com/tensorrt/download/10x

#### Build

```bash
cargo install --path . --features tensorrt
```

```bash
tagger ./assets/sample1_3x1024x1024.webp \
    --devices 0 \
    --v3 vit-large
```

> [!NOTE]
> Currently TensorRT mode is not so fast as CUDA mode.

