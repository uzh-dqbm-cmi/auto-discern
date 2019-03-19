import multiprocessing as mp
import re
from bs4 import BeautifulSoup, Comment, CData, ProcessingInstruction, Declaration, Doctype
from typing import Callable, Dict, List, Tuple


TransformType = Callable[[str], str]


class Transformer:
    """Run a set of transforms on any input. """

    def __init__(self, transforms: List[TransformType], num_cores=8):
        self.transforms = transforms
        self.num_cores = num_cores

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

    def apply(self, input_list: List[Dict]) -> List:
        """Run all transforms on input_list in parallel. """
        pool = mp.Pool(self.num_cores)
        results = pool.map(self._transform_worker, (i for i in input_list))
        pool.close()
        pool.join()
        return results


def remove_tags(soup: BeautifulSoup, tags: List[str]) -> BeautifulSoup:
    """Remove specific tags from the html, including their entire contents."""
    for tag in soup.find_all(True):
        if tag.name in tags:
            # delete tag and its contents
            tag.decompose()
    return soup


def remove_other_xml(soup: BeautifulSoup) -> BeautifulSoup:
    for tag in soup.find_all(string=lambda text: isinstance(text, Comment)
                             or isinstance(text, CData)
                             or isinstance(text, ProcessingInstruction)
                             or isinstance(text, Declaration)
                             or isinstance(text, Doctype)
                             ):
        tag.extract()
    return soup


def remove_html(x: str, replacement_char='. ') -> str:
    """Replace all html tags with replacement_char. """

    soup = BeautifulSoup(x, features="html.parser")
    soup = remove_tags(soup, ['style', 'script'])
    soup = remove_other_xml(soup)
    return soup.get_text(separator=replacement_char)


def remove_html_to_sentences(x: str) -> List[str]:
    """Extract non-html strings as list of strings. """

    soup = BeautifulSoup(x, features="html.parser")
    soup = remove_tags(soup, ['style', 'script'])
    soup = remove_other_xml(soup)
    return [text for text in soup.stripped_strings]


def remove_selected_html(x: str) -> str:
    """Remove all tags except for tags_to_keep, and replace the contents of link tags with LINK"""

    soup = BeautifulSoup(x, features="html.parser")
    soup = remove_tags(soup, ['style', 'script'])
    soup = remove_other_xml(soup)

    tags_to_keep_attr = ['a']
    tags_to_keep = {'a', 'h1', 'h2', 'h3', 'h4', 'br'}
    tags_to_remove = set([tag.name for tag in soup.find_all()]) - tags_to_keep

    for tag in soup.find_all(True):
        if tag.name not in tags_to_keep_attr:
            # clear all attributes
            tag.attrs = {}
        else:
            attrs = dict(tag.attrs)
            for attr in attrs:
                if attr not in ['src', 'href']:
                    del tag.attrs[attr]
                else:
                    tag.attrs[attr] = 'LINK'

    text = str(soup)

    # all tags to remove have been cleared down to their bare tag form without attributes, and can be found/replaced
    for t in tags_to_remove:
        text = text.replace('<{}>'.format(t), '').replace('</{}>'.format(t), '').replace('<{}/>'.format(t), '')

    return text


def replace_problem_chars(x: str, replacement_char=' ') -> str:
    """Replace all problem chars with replacement_char. """
    problem_chars = [
        "\n",
        "\t",
    ]
    for p in problem_chars:
        x = x.replace(p, replacement_char)
    return x


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


def regex_out_periods_and_white_space(text:str) -> str:
    # replaces multiple spaces wth a single space
    text = re.sub(' +', ' ',  text)
    # replaces occurences of '.' followed by any combination of '.', ' ', or '\n' with single '.'
    #  for handling html -> '.' replacement.
    text = re.sub("[.][. \n]{2,}", '.', text)
    return text
