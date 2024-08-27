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

Very experimental and unstable.

#### Prerequisites

cuDDN 9.x **MUST** be installed. You can get it from here:

https://developer.nvidia.com/cudnn-downloads

To build:

```bash
cargo build --features cuda --release
```

Temporary workaround for `libonnxruntime.so.1: cannot open shared object file: No such file or directory` (https://github.com/pykeio/ort/issues/269)

```bash
# find the path of libonnxruntime.so
❯ ls -al ./target/release/libonnxruntime.so
lrwxrwxrwx 1 root root 159  8月 28 03:05 ./target/release/libonnxruntime.so -> /home/USERNAME/.cache/ort.pyke.io/dfbin/x86_64-unknown-linux-gnu/***/onnxruntime/lib/libonnxruntime.

# create a symbolic link
❯ ln -s /home/USERNAME/.cache/ort.pyke.io/dfbin/x86_64-unknown-linux-gnu/***/onnxruntime/lib/libonnxruntime.so \
    ./target/release/libonnxruntime.so.1
```

To run:

```bash
./target/release/tagger ./assets/sample1_3x1024x1024.webp \
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

