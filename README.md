# text2phonemefast: A Python Library for Fast Text to Phoneme Conversion

> **Fork Notice**: This repository is maintained by [Nguyễn Mạnh Cường](https://github.com/manhcuong02) as a fork with enhancements from the original [Text2PhonemeSequence](https://github.com/thelinhbkhn2014/Text2PhonemeSequence) library created by Linh The Nguyen. Thanks to Linh The Nguyen and the co-developers of the project.

This repository is an enhanced and faster version of the original [Text2PhonemeSequence](https://github.com/thelinhbkhn2014/Text2PhonemeSequence) library, which converts text to phoneme sequences for [XPhoneBERT](https://github.com/VinAIResearch/XPhoneBERT).

## Versions

### v0.2.0 (May 12, 2025)
- Language tag support for specified languages
- Case-sensitive handling for acronyms and common words
- Improved processing speed

#### Language Tag Support
Now you can explicitly specify the language for specific words or phrases within mixed-language text using language tags:
- Use `<lang='language_code'>text</lang>` syntax to mark text for specific language processing
- Useful for multilingual texts where automatic language detection may be challenging
- Supports all languages included in the CharsiuG2P dictionary collection

#### Case-Sensitive Processing
The library now distinguishes between:
- Common words in lowercase (e.g., "ba" meaning "father" or "three" in Vietnamese)
- Acronyms in uppercase (e.g., "BA" as an abbreviation)
- This improves pronunciation accuracy in contexts where case carries semantic meaning

### v0.1.0 (April 10, 2025)
- Initial enhanced release with Vietnamese pronunciation fixes
- Performance and architecture improvements
- Support for specialized dictionaries

#### Vietnamese Pronunciation Fixes
- ✅ Fixed "uy" incorrectly pronounced as "ui"
- ✅ Fixed "gì" incorrectly pronounced as "ghì" 
- ✅ Fixed "oo" sound pronunciation
- ✅ Fixed "r", "d", "gi" being pronounced identically
- 🔄 In progress: Fixing "s" and "x" pronounced identically

#### Performance & Architecture Enhancements
- ✅ Applied phoneme post-processing to the dataset inference method (improved consistency)
- ✅ Refactored codebase for better organization and maintainability
- ✅ Created a unique phoneme dictionary per word (instead of segmenting) for improved speed
- ✅ Allow saving words that have never appeared in the G2P dictionary before, so that they do not need to be processed again through the pretrained G2P model, which helps improve speed
- ✅ Merging Vietnamese and English TSV dictionaries for easier multilingual support (Prioritize Vietnamese in case of overlapping sounds, with an estimated 387 overlapping sounds).

## Supported Dictionaries

This library supports several specialized pronunciation dictionaries:

- **Standard dictionaries** - Automatically downloaded from [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P/tree/main/dicts) when needed (e.g., `vie-n.tsv`, `eng-us.tsv`)
- **Enhanced dictionaries** - Specifically optimized for better performance:
  - `vie-n.unique.tsv` - Vietnamese dictionary with optimized pronunciation
  - `eng-us.unique.tsv` - English dictionary with optimized pronunciation
  - `vie-n.mix-eng-us.tsv` - Mixed Vietnamese-English dictionary for multilingual support

When using the `.unique` or `.mix` dictionaries, the library will automatically download them from our repository. These specialized dictionaries provide better pronunciation accuracy, especially for Vietnamese.

## Installation <a name="install"></a>

To install **text2phonemefast**:

```
$ pip install text2phonemefast
```

## Usage Examples <a name="example"></a>

### Basic Usage

This library uses [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P/tree/main) and [segments](https://pypi.org/project/segments/) toolkits for text-to-phoneme conversion. Information about `pretrained_g2p_model` and `language` can be found in the [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P/tree/main) repository.

**Note**: For languages where words are not separated by spaces (e.g., Vietnamese and Chinese), an external tokenizer should be used before feeding the text into the library.

```python
from text2phonemefast import Text2PhonemeFast

# Load Text2PhonemeFast
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
model.infer_sentence("xin chào tôi là Mạnh Cường .")
##Output: "s i n ˧˧ ▁ c a w ˧˨ ▁ t o j ˧˧ ▁ l a ˧˨ ▁ m ɛ ŋ ˨ˀ˩ ʔ ▁ k ɯ ə ŋ ˧˨ ▁ ."
```

### New Features in v0.2.0

#### Using Language Tags

```python
# Process mixed language text with explicit language tags
result = model.infer_sentence("Tôi học <lang='eng-us'>Machine Learning</lang> tại trường đại học .")
print(result)
# Vietnamese words processed with Vietnamese phonemes, "Machine Learning" with English phonemes

# You can also use secondary language dictionaries for better multilingual support
model = Text2PhonemeFast(
    g2p_dict_path="vie-n.unique.tsv",
    language="vie-n",
    secondary_language_dict={
        "eng-us": "eng-us.unique.tsv",
        "fra": "fra.tsv"
    }
)

# Now you can tag French words too
result = model.infer_sentence("Tôi nói <lang='fra'>Bonjour</lang> và <lang='eng-us'>Hello</lang>")
```

#### Case-Sensitive Processing for Acronyms

```python
# Demonstrate the difference between common words and acronyms
result1 = model.infer_sentence("Tôi thấy ba người .")  # "ba" as common word (three)
result2 = model.infer_sentence("Anh ấy làm việc như một BA .")  # "BA" as an acronym
print(f"Common word 'ba': {result1}")
print(f"Acronym 'BA': {result2}")
# The phonetic representation will be different
```

#### Speed Optimization with Missing Phoneme Saving

```python
# First-time processing of text with unknown words
result1 = model.infer_sentence("XPhoneBERT là một mô hình tiên tiến.", save_missing_phonemes=True)
# Now XPhoneBERT is saved to the dictionary

# Subsequent processing will be faster
result2 = model.infer_sentence("XPhoneBERT có hiệu suất tốt.")
# XPhoneBERT is processed from the dictionary instead of the neural model
```

## Credits

This project is a fork of the original work developed by:
- **Linh The Nguyen** - Original author of Text2PhonemeSequence
- **VinAI Research** - Developers of [XPhoneBERT](https://github.com/VinAIResearch/XPhoneBERT)

### Current Maintainer
- **Nguyễn Mạnh Cường** ([manhcuong02](https://github.com/manhcuong02) or [manhcuong17072002](https://github.com/manhcuong17072002)) - Enhanced features and fixes for Vietnamese pronunciation
