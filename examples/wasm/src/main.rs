use ort::Session;
use wdtagger::{pipeline::TaggingPipeline, tagger::TaggerModel};

static MODEL_BYTES: &[u8] = include_bytes!("../models/wd-swinv2-tagger-v3.ort");
const MODEL_NAME: &str = "SmilingWolf/wd-swinv2-tagger-v3";

fn main() -> anyhow::Result<()> {
    ort::wasm::initialize();

    let session = Session::builder()?.commit_from_memory_directly(MODEL_BYTES)?;

    let config = ModelConfig::from_pretrained(&MODEL_NAME)?;
    let preprocessor = ImagePreprocessor::from_config(&config)?;
    let tags = LabelTags::from_pretrained(&MODEL_NAME)?;
    let model = TaggerModel::from_memory(session)?;

    let pipeline = TaggingPipeline::new(model, preprocessor, tags, threshold);

    Ok(())
}
