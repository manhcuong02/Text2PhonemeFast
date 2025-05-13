import os
import re
from typing import Optional, Union, List, Dict, Tuple, Literal

from segments import Tokenizer
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from unidecode import unidecode

# Constants
G2P_DICT_BASE_URL: str = (
    "https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/"
)
CUSTOM_DICT_BASE_URL: str = (
    "https://raw.githubusercontent.com/manhcuong02/Text2PhonemeFast/refs/heads/master/"
)
DEFAULT_SEPARATE_TOKEN: str = "_"
PHONEME_SEPARATOR: str = " ▁ "

# Import from the newly created file
from .language_phoneme_limits import MAX_PHONEME_LENGTHS, DEFAULT_MAX_PHONEME_LENGTH


class Text2PhonemeFast:
    def __init__(
        self,
        pretrained_g2p_model: str = "charsiu/g2p_multilingual_byT5_small_100",
        tokenizer: str = "google/byt5-small",
        language: str = "vie-n",
        g2p_dict_path: Optional[str] = None,
        device: str = "cpu",
        # used for secondary languages specified in language tags
        secondary_language_dict: Dict[str, str] = {},
    ):
        """Initialize the Text2PhonemeFast model.

        This class provides functionality to convert text to phonemes using a pretrained G2P (Grapheme-to-Phoneme) model.
        It supports multiple languages and can handle mixed language text with language tags.

        Args:
            pretrained_g2p_model (str): HuggingFace model path for the G2P model.
                Default: "charsiu/g2p_multilingual_byT5_small_100".
            tokenizer (str): HuggingFace tokenizer path. Default: "google/byt5-small".
            language (str): Primary language code for G2P conversion. Default: "vie-n".
            g2p_dict_path (str, optional): Path to the G2P dictionary file. If None, will
                download the dictionary for the specified language. Default: None.
            device (str): Device to run the model on ("cuda:0", "cpu", etc.). Default: "cuda:0".
            secondary_language_dict (dict): Dictionary mapping language codes to dictionary paths
                for secondary languages support. Default: {}.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_g2p_model)
        self.device = device
        if "cuda" in self.device:
            self.model = self.model.to(self.device)
        self.punctuation = (
            list('.?!,:;-()[]{}<>"') + list("'/‘”“/&#~@^|") + ["...", "*"]
        )
        self.segment_tool = Tokenizer()

        # Initialize G2P dictionary
        self.language, self.g2p_dict_path = self._initialize_g2p_dictionary(
            g2p_dict_path, language
        )

        self.phoneme_dict: Dict[str, Dict[str, List[str]]] = {}
        self.phoneme_dict[self.language] = self.load_g2p(self.g2p_dict_path)

        for sec_language, path in secondary_language_dict.items():
            self.phoneme_dict[sec_language] = self.load_g2p(path)

        self.missing_phonemes: list[dict] = []

    def _initialize_g2p_dictionary(
        self, g2p_dict_path: Optional[str], language: str
    ) -> Tuple[str, str]:
        """Initialize the G2P dictionary for the specified language.

        Args:
            g2p_dict_path (Optional[str]): Path to the G2P dictionary.
            language (str): Language code.

        Returns:
            Tuple[str, str]: A tuple containing (language, g2p_dict_path).

        Raises:
            ValueError: If the language is not supported.
        """
        if g2p_dict_path is None or os.path.exists(g2p_dict_path) is False:
            if g2p_dict_path is not None:
                # Get target directory and filename
                target_dir = os.path.dirname(g2p_dict_path)
                target_file = os.path.basename(g2p_dict_path)
                
                # Create target directory if it doesn't exist and not empty
                if target_dir and not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                
                # Determine download path (using current dir if target_dir is empty)
                download_path = os.path.join(target_dir, target_file) if target_dir else target_file
                
                # Determine the URL source based on file name
                if "unique" in target_file or "mix" in target_file:
                    url_source = CUSTOM_DICT_BASE_URL
                else:
                    url_source = G2P_DICT_BASE_URL
                
                # Download file from appropriate URL
                os.system(f"wget -O {download_path} {url_source}{target_file}")
                
                # Update g2p_dict_path to the full path of the downloaded file
                g2p_dict_path = download_path
            else:
                if os.path.exists("./" + language + ".tsv"):
                    g2p_dict_path = "./" + language + ".tsv"
                else:
                    os.system(f"wget {G2P_DICT_BASE_URL}{language}.tsv")
                    g2p_dict_path = "./" + language + ".tsv"
        else:
            if language is None or len(language) == 0:
                language = g2p_dict_path.split("/")[-1].split(".")[0]

        language_key = f"{language}.tsv"
        if language_key not in MAX_PHONEME_LENGTHS:
            raise ValueError(
                f"Language {language} not supported. Please check the phoneme length dictionary."
            )

        return language, g2p_dict_path

    def save_missing_phonemes(self):
        """
        Save missing phonemes to the G2P dictionary and clear the missing list.

        This function reads the current G2P dictionary file and appends any new phoneme entries
        (i.e., phonemes not already present in the dictionary) to the end of the file.
        After updating the dictionary, it clears the internal list of missing phonemes.

        Args:
            None – This method operates on the instance's attributes, including the G2P dictionary path
            and the list of missing phonemes.

        Returns:
            None – The function performs in-place updates to the G2P dictionary file and internal phoneme state.
        """
        # Open the G2P dictionary file to read content
        with open(self.g2p_dict_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Add missing phonemes to the G2P dictionary
        for item in self.missing_phonemes:
            for text, phoneme in item.items():
                lines.append(f"{text}\t{phoneme}\n")
                print(f"Add new phoneme: {text} -> {phoneme}")

        # Save the new lines to the dictionary file
        with open(self.g2p_dict_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # Clear the list of missing phonemes
        self.missing_phonemes = []

        # load g2p dict again
        self.phoneme_dict[self.language] = self.load_g2p()

    def load_g2p(self, filepath: Optional[str] = None) -> Dict[str, List[str]]:
        """Load a G2P dictionary from file.

        This method loads a grapheme-to-phoneme dictionary from a TSV file where each line
        contains a word and its corresponding phoneme representation separated by a tab.

        Args:
            filepath (str, optional): Path to the G2P dictionary file. If None, uses the
                instance's g2p_dict_path. Default: None.

        Returns:
            dict: A dictionary mapping words (graphemes) to lists of phoneme representations.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If a line in the file does not have exactly two elements.
        """
        if filepath is None:
            filepath = self.g2p_dict_path

        phoneme_dict = {}

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                list_words = f.read().strip().split("\n")

            for word_phone in list_words:
                w_p = word_phone.split("\t")
                if len(w_p) != 2:
                    raise ValueError(
                        f"Invalid format in G2P dictionary line: {word_phone}"
                    )

                if "," not in w_p[1]:
                    phoneme_dict[w_p[0]] = [w_p[1]]
                else:
                    phoneme_dict[w_p[0]] = [w_p[1].split(",")[0]]

            return phoneme_dict
        else:
            raise FileNotFoundError(
                f"G2P dictionary file {filepath} not found. Please check the file path."
            )

    def infer_dataset(
        self,
        input_file="",
        separate_syllable_token="_",
        output_file="",
        save_missing_phonemes=False,
    ):
        """Process a dataset file, converting text to phonemes.

        This method reads an input file where each line contains text data with prefix information,
        converts the text to phonemes, and writes the result to an output file.

        Args:
            input_file (str): Path to the input file containing text data.
            separate_syllable_token (str): Token used to separate syllables in the input text.
                Default: "_".
            output_file (str): Path to save the converted phoneme results.
            save_missing_phonemes (bool): Whether to save any missing phonemes encountered
                during processing to the G2P dictionary. Default: False.

        Returns:
            None: Results are written to the output_file.
        """
        print("Building vocabulary!")

        # Write results to output file
        with open(input_file, "r") as f:
            list_lines = f.readlines()

        with open(output_file, "w") as f:
            for line in tqdm(list_lines):
                line = line.strip().split("|")
                prefix = line[0]
                text = line[-1]

                phonemes = self.infer_sentence(text, separate_syllable_token)

                if len(line) == 3:  # for multi speakers
                    f.write(prefix + "|" + line[1] + "|" + phonemes)
                else:
                    f.write(prefix + "|" + phonemes)
                f.write("\n")

        if save_missing_phonemes:
            self.save_missing_phonemes()

    def get_phoneme_from_dict(self, text: str, language) -> Optional[str]:
        """Look up phoneme for a given text in the G2P dictionary.

        This method searches for the exact text or its lowercase version in the phoneme dictionary
        for the current language. If the text is a punctuation mark, it returns the text itself.

        Args:
            text (str): The text to look up in the phoneme dictionary.

        Returns:
            Optional[str]: The phoneme representation if found, or None if not found.
        """
        if text in self.phoneme_dict[language]:
            return self.phoneme_dict[language][text][0]
        elif text.lower() in self.phoneme_dict[language]:
            return self.phoneme_dict[language][text.lower()][0]
        elif text in self.punctuation:
            return text
        return None

    def text_to_phoneme(
        self,
        text: str,
        language: Optional[str] = None,
        return_type: Literal["string", "list"] = "string",
    ) -> Union[str, List[str]]:
        """Convert text to phoneme representation.

        This method converts the input text to phonemes using either the loaded dictionary
        or the G2P model for words not in the dictionary.

        Args:
            text (str): The text to convert to phonemes.
            language (str, optional): Language code for the text. If None, uses the instance's
                default language. Default: None.
            return_type (str): Format of the returned phonemes, either "string" or "list".
                Default: "string".

        Returns:
            Union[str, List[str]]: Phoneme representation as a single string or a list of strings
                depending on the return_type parameter.
        """
        # Validate input parameters
        if not text:
            return "" if return_type == "string" else []

        if return_type not in ["string", "list"]:
            raise ValueError(
                f"Invalid return_type: {return_type}. Must be 'string' or 'list'"
            )
        if language is None:
            language = self.language
        if language not in self.phoneme_dict:
            self.phoneme_dict[language] = {}

        phoneme = self.get_phoneme_from_dict(text, language)
        if phoneme is not None:
            phones = [phoneme]
        else:
            phones = []
            words = text.split(" ")
            words = [word for word in words if len(word) > 0]
            for word in words:
                phoneme = self.get_phoneme_from_dict(word, language)
                if phoneme is not None:
                    phones.append(phoneme)
                else:
                    out = self.tokenizer(
                        "<" + language + ">: " + word,
                        padding=True,
                        add_special_tokens=False,
                        return_tensors="pt",
                    )
                    if "cuda" in self.device:
                        out["input_ids"] = out["input_ids"].to(self.device)
                        out["attention_mask"] = out["attention_mask"].to(self.device)

                    language_key = language + ".tsv"
                    max_length = MAX_PHONEME_LENGTHS.get(
                        language_key, DEFAULT_MAX_PHONEME_LENGTH
                    )

                    preds = self.model.generate(
                        **out,
                        num_beams=1,
                        max_length=max_length,
                    )
                    phs = self.tokenizer.batch_decode(
                        preds.tolist(), skip_special_tokens=True
                    )

                    phoneme = self.postprocess_phonemes(word, phs[0])

                    self.missing_phonemes.append({word: phoneme})
                    print(f"Missing phoneme: {word} -> {phoneme}")

                    phones.append(phoneme)

        if return_type == "list":
            return phones
        else:
            return "".join(phones)

    def smart_split_with_language_tag(self, text: str) -> list[str]:
        """Split text while preserving language tags.

        This method splits the input text into words but keeps language tag blocks
        intact as single elements in the result list.

        Args:
            text (str): The input text to split, which may contain language tags in
                the format <lang=lang_code>text</lang>.

        Returns:
            list[str]: A list of words and language tag blocks.
        """
        # Regex to capture <lang=...>...</lang> blocks
        lang_pattern = re.compile(r"<lang\s*=.*?>.*?</lang>")

        parts = []
        last_end = 0

        # Process each match
        for match in lang_pattern.finditer(text):
            # Get text before language tag, if any
            before = text[last_end : match.start()]
            if before.strip():
                parts.extend(before.strip().split())

            # Get the entire language tag block
            parts.append(match.group())
            last_end = match.end()

        # Process text after the last match, if any
        after = text[last_end:]
        if after.strip():
            parts.extend(after.strip().split())

        return parts

    def infer_sentence(
        self,
        sentence: str = "",
        separate_syllable_token: str = DEFAULT_SEPARATE_TOKEN,
        save_missing_phonemes: bool = False,
        language: Optional[str] = None,
    ) -> str:
        """Convert a sentence to phoneme representation.

        This method processes a given sentence, handling language tags and converting
        text to phoneme representation. It supports mixed language text by processing
        language tag blocks separately.

        Args:
            sentence (str): The input sentence to convert.
            separate_syllable_token (str): Token used to separate syllables in the input text.
                Default: "_".
            save_missing_phonemes (bool): Whether to save encountered missing phonemes
                to the G2P dictionary. Default: False.
            language (str, optional): Default language for text without explicit language tags.
                If None, uses the instance's default language. Default: None.

        Returns:
            str: The phoneme representation of the input sentence, with phonemes separated
                by the " ▁ " token.
        """
        # Validate input parameters
        if not sentence:
            return ""
        list_words = self.smart_split_with_language_tag(sentence)

        list_phones = []

        for i in range(len(list_words)):
            list_words[i] = list_words[i].replace(separate_syllable_token, " ")

            # normalize apostrophes for english words
            list_words[i] = list_words[i].replace("’", "'")

            if len(list_words[i]) == 0:
                continue

            # extract language from language tag: <lang=eng-us>AI</lang>
            if re.search(r"<lang\s*=(.*?)>", list_words[i]):
                specific_language = re.search(
                    r"<lang\s*=\s*['\"]?(.*?)['\"]?>", list_words[i]
                ).group(1)
                return_type = "list"
                list_words[i] = re.sub(
                    r"<lang\s*=\s*['\"]?(.*?)['\"]?>", "", list_words[i]
                )
                list_words[i] = re.sub(r"</lang>", "", list_words[i])
            else:
                specific_language = language
                return_type = "string"

            phoneme = self.text_to_phoneme(
                list_words[i], language=specific_language, return_type=return_type
            )

            if isinstance(phoneme, str):
                phoneme = [phoneme]
            list_phones.extend(phoneme)

        for i in range(len(list_phones)):
            try:
                segmented_phone = self.segment_tool(list_phones[i], ipa=True)
            except:
                segmented_phone = self.segment_tool(list_phones[i])
            list_phones[i] = segmented_phone

        if save_missing_phonemes:
            self.save_missing_phonemes()

        return " ▁ ".join(list_phones)

    def postprocess_phonemes(self, text: str, phonemes: str) -> str:
        """Apply post-processing rules to generated phonemes.

        This method applies language-specific phoneme transformation rules to improve
        the accuracy of the generated phonemes based on the input text patterns.

        Args:
            text (str): The original input text.
            phonemes (str): The raw phoneme representation generated by the model.

        Returns:
            str: The post-processed phoneme representation.
        """
        phoneme_replacements = {
            r"^(?=.*uy)(?!.*ui).*$": {
                "uj": "wi",
            },
            r"^gi|\sgi($|\s)": {
                "ɣi": "zi",
            },
            r"oo": {"ɔ": "ɔɔ"},
            r"^r": {
                "z": "r",
            },
        }

        # Only used for vie-n.tsv or vie-n.unique.tsv files
        # Not allowed with vie-n.mix-eng-us.tsv
        if "vie" in self.language and "mix" not in self.g2p_dict_path:
            for pattern, replacements in phoneme_replacements.items():
                for t in text.split():
                    match = re.search(pattern, unidecode(t).lower())
                    if match:
                        for key, value in replacements.items():
                            phonemes = phonemes.replace(key, value)

        return phonemes


def demo():
    """Run a demonstration of the Text2PhonemeFast model."""
    model = Text2PhonemeFast(
        g2p_dict_path="vie-n.mix-eng-us.tsv",
        device="cpu",
        language="vie-n",
        secondary_language_dict={
            "eng-us": "eng-us.unique.tsv",
        },
    )

    print(
        model.infer_sentence(
            "Công nghệ AI , <lang='eng-us'>big    data</lang> đang phát triển mạnh mẽ .  ",
            save_missing_phonemes=False,
        )
    )

    print(
        model.infer_sentence(
            "Ba tôi đang là một BA .",
            save_missing_phonemes=False,
        )
    )


if __name__ == "__main__":
    demo()
