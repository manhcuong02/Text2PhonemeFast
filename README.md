# text2phonemefast: A Python Library for Fast Text to Phoneme Conversion

> **Fork Notice**: This repository is maintained by [Nguyễn Mạnh Cường](https://github.com/manhcuong02) as a fork with enhancements from the original [Text2PhonemeSequence](https://github.com/thelinhbkhn2014/Text2PhonemeSequence) library created by Linh The Nguyen.

This repository is an enhanced and faster version of the original [Text2PhonemeSequence](https://github.com/thelinhbkhn2014/Text2PhonemeSequence) library, which converts text to phoneme sequences for [XPhoneBERT](https://github.com/VinAIResearch/XPhoneBERT).

## Key Improvements

### Vietnamese Pronunciation Fixes

- ✅ Fixed "uy" incorrectly pronounced as "ui"
- ✅ Fixed "gì" incorrectly pronounced as "ghì" 
- ✅ Fixed "oo" sound pronunciation
- ✅ Fixed "r", "d", "gi" being pronounced identically
- 🔄 In progress: Fixing "s" and "x" pronounced identically

### Performance & Architecture Enhancements

- ✅ Applied phoneme post-processing to the dataset inference method (improved consistency)
- ✅ Refactored codebase for better organization and maintainability
- ✅ Created a unique phoneme dictionary per word (instead of segmenting) for improved speed
- ✅ Allow saving words that have never appeared in the G2P dictionary before, so that they do not need to be processed again through the pretrained G2P model, which helps improve speed
- ✅ Merging Vietnamese and English TSV dictionaries for easier multilingual support (Prioritize Vietnamese in case of overlapping sounds, with an estimated 405 overlapping sounds).

## Installation <a name="install"></a>

To install **text2phonemefast**:

```
$ pip install text2phonemefast
```

## Usage Examples <a name="example"></a>

This library uses [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P/tree/main) and [segments](https://pypi.org/project/segments/) toolkits for text-to-phoneme conversion. Information about `pretrained_g2p_model` and `language` can be found in the [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P/tree/main) repository.

**Note**: For languages where words are not separated by spaces (e.g., Vietnamese and Chinese), an external tokenizer should be used before feeding the text into the library.

```python
from text2phonemefast import Text2PhonemeFast

# Load Text2PhonemeFast
model = Text2PhonemeFast(pretrained_g2p_model='charsiu/g2p_multilingual_byT5_small_100', language='eng-us', device="cpu")

model = Text2PhonemeFast(
    pretrained_g2p_model='charsiu/g2p_multilingual_byT5_small_100',
    tokenizer="google/byt5-small",
    g2p_dict_path="vie-n.unique.tsv",
    device="cpu", # or cuda
    language="vie-n",
)

# Convert a raw corpus
model.infer_dataset(input_file="/absolute/path/to/input/file", output_file="/absolute/path/to/output/file") 

# Convert a raw sentence
model.infer_sentence("'xin chào tôi là Mạnh Cường .")
##Output: "s i n ˧˧ ▁ c a w ˧˨ ▁ t o j ˧˧ ▁ l a ˧˨ ▁ m ɛ ŋ ˨ˀ˩ ʔ ▁ k ɯ ə ŋ ˧˨ ▁ ."
```

## Credits

This project is a fork of the original work developed by:
- **Linh The Nguyen** - Original author of Text2PhonemeSequence
- **VinAI Research** - Developers of [XPhoneBERT](https://github.com/VinAIResearch/XPhoneBERT)

### Current Maintainer
- **Nguyễn Mạnh Cường** ([manhcuong02](https://github.com/manhcuong02) or [manhcuong17072002](https://github.com/manhcuong17072002)) - Enhanced features and fixes for Vietnamese pronunciation
