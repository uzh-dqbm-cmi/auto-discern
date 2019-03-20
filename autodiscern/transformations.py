import multiprocessing as mp
import re
import spacy
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from bs4 import BeautifulSoup, Comment, CData, ProcessingInstruction, Declaration, Doctype
from typing import Callable, Dict, List, Tuple, Set


TransformType = Callable[[str], str]


class Transformer:
    """Run a set of transforms on any input.
     NOTE: Can run transforms in parallel, but currently not compatible with sentence level segmentation. """

    def __init__(self, leave_some_html: bool = False, segment_into: str = None, parallelism: bool = False, num_cores=8):
        """
        Sets the parameters of the Transformer object.

        Args:
            leave_some_html: bool. Whether or not to leave some html in the text.
            segment_into: str. segment the text into words, sentences, or paragraphs.
            parallelism: bool. Whether to run the transforms in parallel. Not compatible with sentence segmentation.
            num_cores: int. Number of cores to use when using multiprocessing.

        Returns: None

        """
        self.transforms = []
        self.parallelism = parallelism
        self.num_cores = num_cores

        # text cleaning transform to use
        if leave_some_html:
            self.transforms.append(self._to_limited_html)
        else:
            self.transforms.append(self._to_text)

        # segmentation transform to use
        if segment_into is None:
            pass
        elif segment_into == 'words':
            self.transforms.append(self._to_words)
            self.segmentation_type = 'words'
            self.segmenter_helper_obj = WordTokenizer()

        elif segment_into == 'sentences':
            self.transforms.append(self._to_sentences)
            self.segmentation_type = 'sentences'
            nlp = spacy.load('en', disable=['ner'])
            self.segmenter_helper_obj = nlp

        elif segment_into == 'paragraphs':
            self.transforms.append(self._to_paragraphs)
            self.segmentation_type = 'paragraphs'
            self.segmenter_helper_obj = None
        else:
            raise ValueError("Invalid segment_type: {}".format(segment_into))

    def _apply_transforms_to_str(self, content: str) -> str:
        """Apply list all transforms to str. """

        for f in self.transforms:
            content = f(content)
        return content

    def _transform_worker(self, obj: Dict) -> Dict:
        """Create a new transformed object. """
        transformed_obj = {key: obj[key] for key in obj if key != 'content'}
        transformed_obj['content'] = self._apply_transforms_to_str(obj['content'])
        return transformed_obj

    def apply_in_parallel(self, input_list: List[Dict]) -> List:
        """Run all transforms on input_list in parallel. """
        pool = mp.Pool(self.num_cores)
        results = pool.map(self._transform_worker, (i for i in input_list))
        pool.close()
        pool.join()
        return results

    def apply_in_series(self, input_list: List[Dict]) -> List:
        """Run all transforms on input_list in series. """
        results = []
        for i in input_list:
            results.append(self._transform_worker(i))
        return results

    def apply(self, input_list: List[Dict]) -> List:
        """Run all transforms on input_list. """

        if self.parallelism:
            return self.apply_in_parallel(input_list)
        else:
            return self.apply_in_series(input_list)

    # === High Level Transformer functions =======================

    def _to_limited_html(self, x: str) -> str:
        soup = BeautifulSoup(x, features="html.parser")
        soup = self.remove_tags_and_contents(soup, ['style', 'script'])
        soup = self.remove_other_xml(soup)
        soup = self.reformat_html_link_tags(soup)

        tags_to_keep = {'h1', 'h2', 'h3', 'h4'}
        tags_to_keep_with_attr = {'a'}
        tags_to_replace = {
            'br': ('\n', '\n'),
            'p': ('\n', '\n'),
        }
        default_tag_replacement_str = ''
        text = self.replace_html(soup, tags_to_keep, tags_to_keep_with_attr, tags_to_replace,
                                 default_tag_replacement_str)

        text = self.replace_chars(text, ['\t', '\xa0'], ' ')
        text = self.regex_out_periods_and_white_space(text)
        text = self.condense_line_breaks(text)

        return text

    def _to_text(self, x: str) -> str:
        soup = BeautifulSoup(x, features="html.parser")
        soup = self.remove_tags_and_contents(soup, ['style', 'script'])
        soup = self.remove_other_xml(soup)

        tags_to_keep = set()
        tags_to_keep_with_attr = set()
        tags_to_replace = {
            'br': ('\n', '\n'),
            'h1': ('\n', '. \n'),
            'h2': ('\n', '. \n'),
            'h3': ('\n', '. \n'),
            'h4': ('\n', '. \n'),
            'p':  ('\n', '\n'),
        }
        default_tag_replacement_str = ''
        text = self.replace_html(soup, tags_to_keep, tags_to_keep_with_attr, tags_to_replace,
                                 default_tag_replacement_str)

        text = self.replace_chars(text, ['\t', '\xa0'], ' ')
        text = self.regex_out_periods_and_white_space(text)
        text = self.condense_line_breaks(text)

        return text

    # === High Level Segmentation functions ======================

    def _to_words(self, x: str) -> List[str]:
        tok = self.segmenter_helper_obj
        return [str(t) for t in tok.tokenize(x)]

    def _to_sentences(self, x: str) -> List[str]:
        nlp = self.segmenter_helper_obj
        doc = nlp(x)
        result = [sent.string.strip() for sent in doc.sents]
        return result

    def _to_paragraphs(self, x: str) -> List[str]:
        x = self.condense_line_breaks(x)
        return x.split('\n')

    # === BeautifulSoup Helper functions =========================
    @staticmethod
    def remove_tags_and_contents(soup: BeautifulSoup, tags: List[str]) -> BeautifulSoup:
        """Remove specific tags from the html, including their entire contents."""
        for tag in soup.find_all(True):
            if tag.name in tags:
                # delete tag and its contents
                tag.decompose()
        return soup

    @staticmethod
    def remove_other_xml(soup: BeautifulSoup) -> BeautifulSoup:
        for tag in soup.find_all(string=lambda text: isinstance(text, Comment)
                                 or isinstance(text, CData)
                                 or isinstance(text, ProcessingInstruction)
                                 or isinstance(text, Declaration)
                                 or isinstance(text, Doctype)
                                 ):
            tag.extract()
        return soup

    @staticmethod
    def reformat_html_link_tags(soup: BeautifulSoup) -> BeautifulSoup:
        for tag in soup.find_all(True):
            if tag.name == 'a':
                attrs = dict(tag.attrs)
                for attr in attrs:
                    if attr not in ['src', 'href']:
                        del tag.attrs[attr]
                    else:
                        tag.attrs[attr] = 'LINK'
        return soup

    @staticmethod
    def replace_html(soup: BeautifulSoup, tags_to_keep: Set[str], tags_to_keep_with_attr: Set[str],
                     tags_to_replace_with_str: Dict[str, Tuple[str, str]], default_tag_replacement_str: str) -> str:
        """
        Finds all tags in an html BeautifulSoup object and replaces/keeps the tags in accordance with args.

        Args:
            soup: BeautifulSoup object parsing an html
            tags_to_keep: html tags to leave but remove tag attributes
            tags_to_keep_with_attr: html tags to leave entact
            tags_to_replace_with_str: html tags to replace with strings defined in replacement Tuple(start_tag, end_tag)
            default_tag_replacement_str: string to use if no replacement is defined in tags_to_replace_with_str

        Returns: str

        """

        all_tags = set([tag.name for tag in soup.find_all()])
        tags_to_replace = all_tags - tags_to_keep - tags_to_keep_with_attr
        tags_to_replace = tags_to_replace | set(tags_to_replace_with_str.keys())

        for tag in soup.find_all(True):
            if tag.name not in tags_to_keep_with_attr:
                # clear all attributes
                tag.attrs = {}

        text = str(soup)

        # all tags to remove have been cleared down to their bare tag form without attributes, and can be found/replaced
        replacement_tuple = (default_tag_replacement_str, default_tag_replacement_str)
        for tag in tags_to_replace:
            r = tags_to_replace_with_str.get(tag, replacement_tuple)
            text = text.replace('<{}>'.format(tag), r[0]
                                ).replace('</{}>'.format(tag), r[1]
                                          ).replace('<{}/>'.format(tag), r[1])

        return text

    # === String-Based Helper functions =====-====================

    @staticmethod
    def regex_out_periods_and_white_space(text: str) -> str:
        # replaces multiple spaces wth a single space
        text = re.sub(' +', ' ',  text)
        # replace occurences of '.' followed by any combination of '.', ' ', or '\n' with single '.'
        #  for handling html -> '.' replacement.
        text = re.sub("[.][. ]{2,}", '. ', text)
        text = re.sub("[.][. \n]{2,}", '. \n', text)
        return text

    @staticmethod
    def condense_line_breaks(text: str) -> str:
        # replaces multiple spaces wth a single space
        text = re.sub(r' +', ' ',  text).strip()

        # replace html line breaks with new line characters
        text = re.sub(r'<br[/]*>', '\n', text)

        # replace any combination of ' ' and '\n' with single ' \n'
        text = re.sub(r"[ \n]{2,}", ' \n', text)
        return text

    @staticmethod
    def replace_chars(x: str, chars_to_replace: List[str], replacement_char: str) -> str:
        """Replace all chars_to_replace with replacement_char. """
        for p in chars_to_replace:
            x = x.replace(p, replacement_char)
        return x


# ============================================================
# === Other, not yet integrated into the class yet ===========
# ============================================================

def allennlp_ner_tagger(sentence: str, predictor: Callable) -> List[Tuple[str, str]]:
    # pass this function the predictor of
    # predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
    prediction = predictor.predict(sentence)
    # the predictor also gives logits. For now we just want to look at the tags
    return [(w, prediction['tags'][i]) for i, w in enumerate(prediction['words'])]


def ner_tuples_to_html(tuples: List[Tuple[str, str]]) -> str:
    """Display the output of allennlp_ner_tagger as text color-coded by ner type.
    Wrap this function call in IPython.display.HTML() to see output in notebook. """

    ner_type_to_html_tag = {
        "U-PER": 'font  color="blue"',
        "B-ORG": 'font  color="green"',
        "L-ORG": 'font  color="red"',
        "U-MISC": 'font  color="orange"',
    }

    ner_html = ""
    for sentence in tuples:
        for token in sentence:
            text = token[0]
            ner_type = token[1]
            if ner_type == 'O':
                ner_html += " {} ".format(text)
            else:
                tag = ner_type_to_html_tag[ner_type]
                ner_html += " <{0}>{1}</{0}>".format(tag, text)

    return ner_html
