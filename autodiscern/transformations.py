import html
import multiprocessing as mp
import re
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from bs4 import BeautifulSoup, Comment, CData, ProcessingInstruction, Declaration, Doctype
from nltk.tokenize.punkt import PunktSentenceTokenizer
from typing import Callable, Dict, List, Tuple, Set


TransformType = Callable[[str], str]


class Transformer:
    """Run a set of transforms and annotations on any input.

    Transforms are run on the string level. A transform takes a string as input, and returns a modified string.

    Segmentation is run as a special transform that takes a string and returns a List(str). Segmentation is always
        run after all other transforms.

    Annotations are run on the dict level. An annotator takes a dict with the ['content'] key set to a string, and
        returns a modified dict with additional keys. The annotation may also modify the string in ['content'].
    """

    def __init__(self, leave_some_html: bool = False, html_to_plain_text: bool = False, segment_into: str = None,
                 flatten: bool = False, annotate_html: bool = False, parallelism: bool = False, num_cores=8):
        """
        Sets the parameters of the Transformer object.

        Args:
            leave_some_html: bool. Whether or not to leave some html in the text.
            html_to_plain_text: bool. Convert html tags into plain text so as not to break text segmentation.

            segment_into: str. segment the text into words, sentences, or paragraphs.
            flatten: bool. Whether to flatten the segmented texts list(dict(list)) into a single master list(dict).

            annotate_html: bool. Set annotations on the dict about the presence of html tags. Also removes html tags.

            parallelism: bool. Whether to run the transforms in parallel. Not compatible with sentence segmentation.
            num_cores: int. Number of cores to use when using multiprocessing.

        Returns: None

        """
        self.transforms = []
        self.annotations = []
        self.parallelism = parallelism
        self.num_cores = num_cores
        self.flatten = flatten

        if leave_some_html and segment_into is not None and html_to_plain_text is False:
            print("WARNING: segmentation does not work well with html remaining in the sentence. Consider setting "
                  "html_to_plain_text to True. ")

        if flatten and segment_into is None:
            print("WARNING: Flattening should only be applied when segmentation is also applied. Make sur you know "
                  "what you're doing! ")

        # text cleaning transform to use
        if leave_some_html:
            if html_to_plain_text:
                self.transforms.append(self._to_limited_html_plain_text)
            else:
                self.transforms.append(self._to_limited_html)
        else:
            self.transforms.append(self._to_text)

        # segmentation transform to use
        if segment_into is None:
            pass
        elif segment_into in {'w', 'word', 'words'}:
            self.transforms.append(self._to_words)
            self.segmentation_type = 'words'
            self.segmenter_helper_obj = WordTokenizer()
        elif segment_into in {'s', 'sent', 'sents', 'sentence', 'sentences'}:
            self.transforms.append(self._to_sentences)
            self.segmentation_type = 'sentences'
            # nlp = spacy.load('en', disable=['ner'])
            # self.segmenter_helper_obj = nlp
            self.segmenter_helper_obj = PunktSentenceTokenizer()
        elif segment_into in {'p', 'para', 'paragraph', 'paragraphs'}:
            self.transforms.append(self._to_paragraphs)
            self.segmentation_type = 'paragraphs'
            self.segmenter_helper_obj = None
        else:
            raise ValueError("Invalid segment_type: {}".format(segment_into))

        # annotations to use
        if annotate_html:
            self.annotations.append(self._annotate_and_clean_html)

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

    def _apply_annotations_to_dict(self, content: Dict) -> Dict:
        """Apply list all annotations to dict. """

        for f in self.annotations:
            content = f(content)
        return content

    def _annotation_worker(self, obj: Dict) -> Dict:
        """Create a new transformed object. """
        return self._apply_annotations_to_dict(obj)

    def apply_in_parallel(self, input_list: List[Dict], worker: Callable) -> List:
        """Run all transforms on input_list in parallel. """
        pool = mp.Pool(self.num_cores)
        results = pool.map(worker, (i for i in input_list))
        pool.close()
        pool.join()
        return results

    def apply_in_series(self, input_list: List[Dict], worker: Callable) -> List:
        """Run all transforms on input_list in series. """
        results = []
        for i in input_list:
            results.append(worker(i))
        return results

    def _apply_transforms(self, input_list: List[Dict]) -> List:
        if self.parallelism:
            result = self.apply_in_parallel(input_list, worker=self._transform_worker)
        else:
            result = self.apply_in_series(input_list, worker=self._transform_worker)
        return result

    def _apply_annotations(self, input_list: List[Dict]) -> List:
        if self.parallelism:
            result = self.apply_in_parallel(input_list, worker=self._annotation_worker)
        else:
            result = self.apply_in_series(input_list, worker=self._annotation_worker)
        return result

    def apply(self, input_list: List[Dict]) -> List:
        """Run all transforms and annotations on input_list. """

        # run transformations, including segmentation, at the string level
        result = self._apply_transforms(input_list)

        # if segmentation occurred, flatten list of dicts with now lists in 'content' into one single list of dicts
        if self.flatten:
            result = self._flatten_text_dicts(result)

        # run annotations, at the dict level
        result = self._apply_annotations(result)

        return result

    # === High Level Transformer functions =======================

    def _to_limited_html(self, x: str) -> str:
        soup = BeautifulSoup(x, features="html.parser")
        soup = self.remove_tags_and_contents(soup, ['style', 'script'])
        soup = self.remove_other_xml(soup)
        soup = self.reformat_html_link_tags(soup)

        tags_to_keep = {'h1', 'h2', 'h3', 'h4'}
        tags_to_keep_with_attr = {'a'}
        tags_to_replace = {
            'br': ('.\n', '.\n'),
            'p': ('\n', '\n'),
        }
        default_tag_replacement_str = ''
        text = self.replace_html(soup, tags_to_keep, tags_to_keep_with_attr, tags_to_replace,
                                 default_tag_replacement_str)

        text = self.replace_chars(text, ['\t', '\xa0'], ' ')
        text = self.regex_out_punctuation_and_white_space(text)
        text = self.condense_line_breaks(text)

        return text

    def _to_limited_html_plain_text(self, x: str) -> str:
        soup = BeautifulSoup(x, features="html.parser")
        soup = self.remove_tags_and_contents(soup, ['style', 'script'])
        soup = self.remove_other_xml(soup)
        soup = self.reformat_html_link_tags(soup)

        tags_to_keep = set()
        tags_to_keep_with_attr = set()
        tags_to_replace = {
            'br': ('.\n', '.\n'),
            'h1': (' thisisah1tag ', '. \n'),
            'h2': (' thisisah2tag ', '. \n'),
            'h3': (' thisisah3tag ', '. \n'),
            'h4': (' thisisah4tag ', '. \n'),
            'a':  (' thisisalinktag ', ' '),
            'li': ('thisisalistitemtag ', '. '),
            'tr': ('thisisatablerowtag ', '. '),
            'p':  ('\n', '\n'),
        }
        default_tag_replacement_str = ''
        text = self.replace_html(soup, tags_to_keep, tags_to_keep_with_attr, tags_to_replace,
                                 default_tag_replacement_str)

        text = self.replace_chars(text, ['\t', '\xa0'], ' ')
        text = self.regex_out_punctuation_and_white_space(text)
        text = self.condense_line_breaks(text)

        return text

    def _to_text(self, x: str) -> str:
        soup = BeautifulSoup(x, features="html.parser")
        soup = self.remove_tags_and_contents(soup, ['style', 'script'])
        soup = self.remove_other_xml(soup)

        tags_to_keep = set()
        tags_to_keep_with_attr = set()
        tags_to_replace = {
            'br': ('.\n', '.\n'),
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
        text = self.regex_out_punctuation_and_white_space(text)
        text = self.condense_line_breaks(text)

        return text

    # === High Level Segmentation functions ======================

    def _to_words(self, x: str) -> List[str]:
        tok = self.segmenter_helper_obj
        return [str(t) for t in tok.tokenize(x)]

    def _to_sentences(self, x: str) -> List[str]:
        # spacy sentence tokenizer version
        # nlp = self.segmenter_helper_obj
        # doc = nlp(x)
        # result = [sent.string.strip() for sent in doc.sents]
        tokenizer = self.segmenter_helper_obj
        result = [sent.strip() for sent in tokenizer.tokenize(x)]
        return result

    def _to_paragraphs(self, x: str) -> List[str]:
        x = self.condense_line_breaks(x)
        return x.split('\n')

    @staticmethod
    def _flatten_text_dicts(list_of_dicts: List[Dict]) -> List[Dict]:
        output = []
        for parent_dict in list_of_dicts:
            for num, i in enumerate(parent_dict['content']):
                child_dict = {key: parent_dict[key] for key in parent_dict if key != 'content'}
                child_dict['content'] = i
                child_dict['sub_id'] = num
                output.append(child_dict)
        return output

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

    def replace_html(self, soup: BeautifulSoup, tags_to_keep: Set[str], tags_to_keep_with_attr: Set[str],
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

        text = self.soup_to_text_with_tags(soup)

        # all tags to remove have been cleared down to their bare tag form without attributes, and can be found/replaced
        replacement_tuple = (default_tag_replacement_str, default_tag_replacement_str)
        for tag in tags_to_replace:
            r = tags_to_replace_with_str.get(tag, replacement_tuple)
            text = text.replace('<{}>'.format(tag), r[0]
                                ).replace('</{}>'.format(tag), r[1]
                                          ).replace('<{}/>'.format(tag), r[1])

        return text

    @staticmethod
    def soup_to_text_with_tags(soup: BeautifulSoup) -> str:
        text = str(soup)
        text = html.unescape(text)
        return text

    # === String-Based Helper functions ==========================

    @staticmethod
    def regex_out_punctuation_and_white_space(text: str) -> str:
        text = text.replace('?.', '?')

        # replaces multiple spaces wth a single space
        text = re.sub(' +', ' ',  text)
        # replace occurences of '.' followed by any combination of '.', ' ', or '\n' with single '.'
        #  for handling html -> '.' replacement.
        text = re.sub("[.][. ]{2,}", '. ', text)
        text = re.sub("[.][. \n]{2,}", '. \n', text)

        # if there is a period at the very start of the document, remove it (replace 1 time)
        text = text.lstrip()
        if len(text) > 0 and text[0] == '.':
            text = text.replace('.', '', 1).lstrip()

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

    # === Annotation Functions ==================================

    @staticmethod
    def _annotate_and_clean_html(d: Dict) -> Dict:
        tags = {
            'thisisah1tag': 'h1',
            'thisisah2tag': 'h2',
            'thisisah3tag': 'h3',
            'thisisah4tag': 'h4',
            'thisisalinktag': 'a',
            'thisisalistitemtag': 'li',
            'thisisatablerowtag': 'tr',
        }
        found_tags = []
        for plaintexttag in tags:
            if plaintexttag in d['content']:
                d['content'] = d['content'].replace(plaintexttag, ' ').strip()
                found_tags.append(tags[plaintexttag])
        d['html_tags'] = found_tags
        return d


# ============================================================
# === Other non-class functions ==============================
# ============================================================

def get_id(d: Dict) -> str:
    identifier = d['id']
    if 'sub_id' in d:
        identifier = "{}-{}".format(d['id'], d['sub_id'])
    return identifier


def convert_list_of_dicts_to_dict_of_dicts(input_list: List[Dict]) -> Dict[str, Dict]:
    output_dict = {}
    for d in input_list:
        id = get_id(d)
        output_dict[id] = d
    return output_dict


def add_metamap_annotations(inputs: Dict[str, Dict], metamap_path: str = None) -> Dict[str, Dict]:
    from pymetamap import MetaMapLite

    if metamap_path is None:
        print("NOTE: no metamap path provided. Using Laura's default")
        metamap_path = '/Users/laurakinkead/Documents/metamap/public_mm_lite/'

    # create list of ids, and list of sentences in matching order
    ids = inputs.keys()
    sentences = []
    for id in ids:
        sentences.append(inputs[id]['content'])

    # run metamap
    mm = MetaMapLite.get_instance(metamap_path)
    concepts, error = mm.extract_concepts(sentences, ids)

    # add concepts with score above 1 to input sentences
    for concept in concepts:
        concept_dict = {}
        if float(concept.score) > 1:
            for fld in concept._fields:
                concept_dict[fld] = getattr(concept, fld)

            # attach concept to input_dict
            id = concept_dict['index']
            id = id.replace('"', '').replace("'", '')
            if 'metamap' not in inputs[id]:
                inputs[id]['metamap'] = []
            inputs[id]['metamap'].append(concept_dict)

    return inputs


def add_ner_annotations(inputs: Dict[str, Dict]) -> Dict[str, Dict]:
    # is there a batch predictor?
    # https://allenai.github.io/allennlp-docs/api/allennlp.predictors.html#sentence-tagger

    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")

    for id in inputs:
        sentence = inputs[id]['content']
        ner_output = allennlp_ner_tagger(sentence, predictor)
        ner_output = [x for x in ner_output if x[1] != 'O']
        inputs[id]['ner'] = ner_output
    return inputs


def allennlp_ner_tagger(sentence: str, predictor: Callable) -> List[Tuple[str, str]]:
    # pass this function the predictor of
    # predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
    if len(sentence) > 1:
        prediction = predictor.predict(sentence)
        # the predictor also gives logits. For now we just want to look at the tags
        return [(w, prediction['tags'][i]) for i, w in enumerate(prediction['words'])]
    else:
        return []


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
