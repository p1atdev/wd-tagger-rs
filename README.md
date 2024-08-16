# wd-tagger-rs

An inference tool of [WaifuDiffusion Tagger](https://huggingface.co/spaces/SmilingWolf/wd-tagger).

> [!IMPORTANT]
> WIP. Not ready for use.

## Usage (Experimental)

(Only tested on Ubuntu 24.04)

> [!WARNING]
> `cargo install` is not working for some reason. please help.

You need Rust and Cargo to build.

To build:

```bash
git clone https://github.com/p1atdev/wd-tagger-rs
cd wd-tagger-rs
cargo build --release
```

To run:

```bash
./target/release/tagger ./assets/sample1_3x1024x1024.webp
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

