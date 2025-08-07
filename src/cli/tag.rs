#[rustfmt::skip]
pub const UNDERSCORE_TAGS: [&str; 19] = [
    ">_<",
    ">_o",
    "0_0",
    "o_o",
    "3_3",
    "6_9",
    "@_@",
    "u_u",
    "x_x",
    "^_^",
    "|_|",
    "=_=",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    // ↓ deprecated
    "||_||",
    "(o)_(o)",
];

pub fn fix_tag_underscore(tag: &str) -> String {
    if UNDERSCORE_TAGS.contains(&tag) {
        tag.to_string()
    } else {
        tag.replace('_', " ")
    }
}
