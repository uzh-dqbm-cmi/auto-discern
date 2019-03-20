import spacy
from typing import List

# ============================================================
# === High Level Segmentation functions ======================
# ============================================================


class Segmenter:

    def __init__(self, segment_type: str) -> None:
        if segment_type == 'words':
            self.segmentation_type = 'words'
            from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
            self.segmenter_helper_obj = WordTokenizer()

        elif segment_type == 'sentences':
            self.segmentation_type = 'sentences'
            nlp = spacy.load('en_core_web_sm')
            self.segmenter_helper_obj = nlp

        elif segment_type == 'paragraphs':
            self.segmentation_type = 'paragraphs'
            self.segmenter_helper_obj = None

        else:
            raise ValueError("Invalid segment_type: {}".format(segment_type))

    def segment(self, x: List[str]) -> List[List[str]]:
        if self.segmentation_type == 'words':
            return self._to_words(x, self.segmenter_helper_obj)
        elif self.segmentation_type == 'sentences':
            return self._to_words(x, self.segmenter_helper_obj)
        elif self.segmentation_type == 'paragraphs':
            return self._to_words(x)

    @staticmethod
    def _to_words(x: List[str], tok) -> List[List[str]]:
        return tok.batch_tokenize(x)

    @staticmethod
    def _to_sentences(x: List[str], nlp) -> List[List[str]]:
        result = []
        for t in x:
            doc = nlp(t)
            result.append([sent for sent in doc.sents])
        return result

    @staticmethod
    def _to_paragraphs(x: List[str]) -> List[List[str]]:
        return [t.split('\n') for t in x]
