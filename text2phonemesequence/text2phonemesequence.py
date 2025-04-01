import os
import re

from segments import Tokenizer
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration
from unidecode import unidecode


class Text2PhonemeSequence:
    def __init__(
        self,
        pretrained_g2p_model="charsiu/g2p_multilingual_byT5_small_100",
        tokenizer="google/byt5-small",
        language="vie-n",
        g2p_dict_path=None,
        device="cuda:0",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_g2p_model)
        self.device = device
        if "cuda" in self.device:
            self.model = self.model.to(self.device)
        self.punctuation = (
            list('.?!,:;-()[]{}<>"') + list("'/‘”“/&#~@^|") + ["...", "*"]
        )
        self.segment_tool = Tokenizer()
        self.language = language
        self.phoneme_length = {
            "msa.tsv": 27,
            "amh.tsv": 28,
            "urd.tsv": 152,
            "mri.tsv": 30,
            "glg.tsv": 22,
            "swa.tsv": 28,
            "est.tsv": 47,
            "hbs-latn.tsv": 31,
            "spa.tsv": 28,
            "pol.tsv": 38,
            "spa-latin.tsv": 24,
            "fra.tsv": 43,
            "uig.tsv": 33,
            "aze.tsv": 34,
            "nob.tsv": 52,
            "swe.tsv": 36,
            "por-bz.tsv": 28,
            "arg.tsv": 30,
            "yue.tsv": 227,
            "sqi.tsv": 42,
            "cat.tsv": 27,
            "hbs-cyrl.tsv": 31,
            "grc.tsv": 39,
            "bak.tsv": 44,
            "hun.tsv": 67,
            "lat-clas.tsv": 61,
            "ita.tsv": 37,
            "san.tsv": 38,
            "arm-w.tsv": 41,
            "afr.tsv": 81,
            "ind.tsv": 87,
            "sme.tsv": 31,
            "egy.tsv": 22,
            "rus.tsv": 62,
            "ady.tsv": 29,
            "epo.tsv": 43,
            "srp.tsv": 42,
            "zho-t.tsv": 135,
            "tha.tsv": 295,
            "vie-c.tsv": 184,
            "kur.tsv": 37,
            "ice.tsv": 37,
            "geo.tsv": 75,
            "cze.tsv": 43,
            "ara.tsv": 370,
            "tgl.tsv": 27,
            "nan.tsv": 175,
            "por-po.tsv": 35,
            "bos.tsv": 52,
            "enm.tsv": 24,
            "ltz.tsv": 55,
            "ina.tsv": 26,
            "ukr.tsv": 39,
            "mac.tsv": 44,
            "kaz.tsv": 44,
            "slk.tsv": 73,
            "ang.tsv": 33,
            "khm.tsv": 47,
            "syc.tsv": 24,
            "eus.tsv": 35,
            "spa-me.tsv": 28,
            "tts.tsv": 26,
            "gle.tsv": 36,
            "slv.tsv": 35,
            "ido.tsv": 23,
            "zho-s.tsv": 135,
            "vie-n.tsv": 174,
            "gre.tsv": 20,
            "eng-us.tsv": 119,
            "fin.tsv": 44,
            "pap.tsv": 26,
            "tuk.tsv": 49,
            "jpn.tsv": 174,
            "bel.tsv": 58,
            "uzb.tsv": 36,
            "dan.tsv": 42,
            "ori.tsv": 45,
            "ron.tsv": 30,
            "bul.tsv": 40,
            "bur.tsv": 52,
            "lit.tsv": 87,
            "dut.tsv": 48,
            "fra-qu.tsv": 255,
            "eng-uk.tsv": 129,
            "hin.tsv": 110,
            "isl.tsv": 41,
            "arm-e.tsv": 38,
            "kor.tsv": 175,
            "tur.tsv": 54,
            "fas.tsv": 70,
            "ger.tsv": 56,
            "tam.tsv": 38,
            "wel-sw.tsv": 44,
            "vie-s.tsv": 172,
            "mlt.tsv": 47,
            "wel-nw.tsv": 49,
            "slo.tsv": 27,
            "lat-eccl.tsv": 43,
            "snd.tsv": 60,
            "tat.tsv": 103,
            "alb.tsv": 40,
            "hau.tsv": 45,
            "pus.tsv": 34,
            "tib.tsv": 61,
            "sga.tsv": 47,
            "heb.tsv": 39,
            "hrx.tsv": 27,
            "fao.tsv": 42,
            "dsb.tsv": 29,
        }
        self.phone_dict = {}
        if g2p_dict_path is None:
            os.system(
                "wget https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/"
                + language
                + ".tsv"
            )
            g2p_dict_path = "./" + language + ".tsv"

        if os.path.exists(g2p_dict_path):
            f = open(g2p_dict_path, "r", encoding="utf-8")
            list_words = f.read().strip().split("\n")
            f.close()
            for word_phone in list_words:
                w_p = word_phone.split("\t")
                assert len(w_p) == 2, print(w_p)
                if "," not in w_p[1]:
                    self.phone_dict[w_p[0]] = [w_p[1]]
                else:
                    self.phone_dict[w_p[0]] = [w_p[1].split(",")[0]]

    def infer_dataset(
        self,
        input_file="",
        seperate_syllabel_token="_",
        output_file="",
        batch_size=64,
    ):
        f = open(input_file, "r")
        list_lines = f.readlines()
        f.close()
        list_words = []
        print("Building vocabulary!")
        for line in list_lines:
            words = line.strip().split("|")[-1].split(" ")
            for w in words:
                w = w.replace(seperate_syllabel_token, " ").lower()
                if w not in self.phone_dict.keys():
                    list_words.append(w)
        list_words_p = ["<" + self.language + ">: " + i for i in list_words]
        list_words_p_batch = []
        list_words_batch = []
        temp_list = []
        temp_list_raw = []
        for m in range(len(list_words_p)):
            temp_list.append(list_words_p[m])
            temp_list_raw.append(list_words[m])
            if len(temp_list) == batch_size:
                list_words_p_batch.append(temp_list)
                temp_list = []
                list_words_batch.append(temp_list_raw)
                temp_list_raw = []
        if len(temp_list) != 0:
            list_words_p_batch.append(temp_list)
            list_words_batch.append(temp_list_raw)

        for j in range(len(list_words_p_batch)):
            out = self.tokenizer(
                list_words_p_batch[j],
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            if "cuda" in self.device:
                out["input_ids"] = out["input_ids"].to(self.device)
                out["attention_mask"] = out["attention_mask"].to(self.device)
            if self.language + ".tsv" not in self.phoneme_length.keys():
                self.phoneme_length[self.language + ".tsv"] = 50
            preds = self.model.generate(
                **out,
                num_beams=1,
                max_length=self.phoneme_length[self.language + ".tsv"]
            )
            phones = self.tokenizer.batch_decode(
                preds.tolist(), skip_special_tokens=True
            )
            assert len(phones) == len(list_words_p_batch[j])

            for i in range(len(phones)):
                if list_words_batch[j][i] in self.punctuation:
                    phones[i] = list_words_batch[j][i]
                # Áp dụng postprocess_phonemes cho phoneme
                if "vie" in self.language:
                    phones[i] = self.postprocess_phonemes(list_words_batch[j][i], phones[i])
                self.phone_dict[list_words_batch[j][i]] = [phones[i]]
        for w in self.phone_dict.keys():
            try:
                segmented_phone = self.segment_tool(self.phone_dict[w][0], ipa=True)
            except:
                segmented_phone = self.segment_tool(self.phone_dict[w][0])
            self.phone_dict[w].append(segmented_phone)

        f = open(input_file, "r")
        list_lines = f.readlines()
        f.close()
        f = open(output_file, "w")
        for line in tqdm(list_lines):
            line = line.strip().split("|")
            prefix = line[0]
            list_words = line[-1].split(" ")
            for i in range(len(list_words)):
                list_words[i] = self.phone_dict[
                    list_words[i].replace(seperate_syllabel_token, " ").lower()
                ][1]
            if len(line) == 3: # for multi speakers
                f.write(prefix + "|" + line[1] + "|" + " ▁ ".join(list_words))
            else:
                f.write(prefix + "|" + " ▁ ".join(list_words))
            f.write("\n")
        f.close()

    def t2p(self, text: str) -> str:
        if text in self.phone_dict:
            return self.phone_dict[text][0]
        elif text in self.punctuation:
            return text
        else:
            out = self.tokenizer(
                "<" + self.language + ">: " + text,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            if "cuda" in self.device:
                out["input_ids"] = out["input_ids"].to(self.device)
                out["attention_mask"] = out["attention_mask"].to(self.device)
            if self.language + ".tsv" not in self.phoneme_length.keys():
                self.phoneme_length[self.language + ".tsv"] = 50
            preds = self.model.generate(
                **out,
                num_beams=1,
                max_length=self.phoneme_length[self.language + ".tsv"]
            )
            phones = self.tokenizer.batch_decode(
                preds.tolist(), skip_special_tokens=True
            )
            return phones[0]

    def infer_sentence(self, sentence="", seperate_syllabel_token="_"):
        list_words = sentence.split(" ")
        list_phones = []
        for i in range(len(list_words)):
            list_words[i] = list_words[i].replace(seperate_syllabel_token, " ")
            phoneme = self.t2p(list_words[i])
            list_phones.append(phoneme)

            list_phones[-1] = self.postprocess_phonemes(list_words[i], list_phones[-1])

        for i in range(len(list_phones)):
            try:
                segmented_phone = self.segment_tool(list_phones[i], ipa=True)
            except:
                segmented_phone = self.segment_tool(list_phones[i])
            list_phones[i] = segmented_phone
        return " ▁ ".join(list_phones)

    def postprocess_phonemes(self, text: str, phonemes: str) -> str:
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

        if "vie" in self.language:
            for pattern, replacements in phoneme_replacements.items():
                for t in text.split():
                    match = re.search(pattern, unidecode(t).lower())
                    if match:
                        old_phoneme: str = self.t2p(t)
                        new_phoneme = old_phoneme
                        for key, value in replacements.items():
                            new_phoneme = new_phoneme.replace(key, value)

                        phonemes = phonemes.replace(old_phoneme, new_phoneme)

        return phonemes


if __name__ == "__main__":

    model = Text2PhonemeSequence(
        g2p_dict_path = "vie.tsv",
        device = "cpu",
    )
    model.infer_dataset(
        input_file="text2phonemesequence/input.txt",
        output_file="text2phonemesequence/output.txt",
        batch_size=64,
    )
