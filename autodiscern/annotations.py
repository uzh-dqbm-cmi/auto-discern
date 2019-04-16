from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.predictors.predictor import Predictor
from bs4 import BeautifulSoup
from bs4.element import Tag
from typing import Callable, Dict, List, Tuple
from autodiscern.data_manager import DataManager


def add_word_token_annotations(inputs: Dict[str, Dict]) -> Dict[str, Dict]:
    tok = WordTokenizer()
    for id in inputs:
        inputs[id]['tokens'] = tok.tokenize(inputs[id]['content'])
    return inputs


def add_metamap_annotations(inputs: Dict[str, Dict], dm: DataManager, metamap_path: str = None) -> Dict[str, Dict]:
    from pymetamap import MetaMapLite

    if metamap_path is None:
        print("NOTE: no metamap path provided. Using Laura's default")
        metamap_path = '/Users/laurakinkead/Documents/metamap/public_mm_lite/'

    metamap_semantics = dm.metamap_semantics
    groups_to_keep = [
        'Anatomy',
        'Devices',
        'Disorders',
        'Physiology',
        'Procedures',
    ]
    types_to_keep = [
        # Chemicals & Drugs
        'Antibiotic',
        'Clincal Drug',
        'Enzyme',
        'Hormone',
        'Pharmacologic Substance',
        'Receptor',
    ]
    semantic_type_filter_df = metamap_semantics[metamap_semantics['group_name'].isin(groups_to_keep) |
                                                metamap_semantics['name'].isin(types_to_keep)
                                                ].sort_values(['group_name', 'name'])
    semantic_type_filter = list(semantic_type_filter_df['abbreviation'])

    # create list of ids, and list of sentences in matching order
    ids = inputs.keys()
    sentences = []
    for id in ids:
        sentences.append(inputs[id]['content'])

    # run metamap
    mm = MetaMapLite.get_instance(metamap_path)
    concepts, error = mm.extract_concepts(sentences, ids)

    # add concepts with score above 1 and matching the semantics filter to input sentences
    for concept in concepts:
        concept_dict = {}
        if float(concept.score) > 1:
            semtypes = concept.semtypes.replace('[', '').replace(']', '').split(',')
            for semtype in semtypes:
                if semtype in semantic_type_filter:
                    for fld in concept._fields:
                        concept_dict[fld] = getattr(concept, fld)
                        break

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


def extract_potential_references(text: str) -> List[str]:
    """
    Find the last-most instance of a reference keywords in an html heading, and retrieve all text following the heading.
    Args:
        text: str. HTML of webpage.

    Returns: List[str]. List of potential citations found under the heading, split into a list based on line breaks.

    """
    soup = BeautifulSoup(text, features="html.parser")

    reference_keywords = ['references', 'citations', 'bibliography']

    # iterate backwards through all headers
    header_tags = soup.find_all(['h1', 'h2', 'h3', 'h4'])
    for tag in header_tags[-1:]:
        # if any single word in header matches a ref keyword
        if tag.string is not None and any(h in tag.string.lower().split(' ') for h in reference_keywords):
            # return the remainder of the document
            potential_citations = []
            for sibling_tag in tag.next_siblings:
                if type(sibling_tag) == Tag:
                    sibling_text = sibling_tag.get_text()
                    text_split = [line for line in sibling_text.split('\n') if line.strip() != '']
                    potential_citations.extend(text_split)
            return potential_citations
    return []


def annotate_potential_references(potential_references: List[str]) -> List[Tuple[str, List[str]]]:
    """Placeholder function for annotating candidate reference strings.
    TODO: use NeuralParsCit here
    """
    annotated_references = []
    for item in potential_references:
        annotated_references.append((item, ['token', 'fake_annotation']))
    return annotated_references


def evaluate_potential_references(potential_references: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
    """Select references which meet requirements based on the reference's token-based annotations.
    Only keep references that have at least one of all (author, title).

    Args:
            potential_references: List of references and their annotations represented by tuples, like so:
                [
                    ('original_reference_string', [('token1', 'annotation1'), ('token2', 'annotation2'), ...],
                    ...
                ]

    Returns: A subset of potential_references with the same data structure.
    """
    selected_references = []
    for orig_ref_string, ref_annotations in potential_references:
        annotation_types = set([annotation for token, annotation in ref_annotations])
        if 'title' in annotation_types and 'author' in annotation_types:
            selected_references.append((orig_ref_string, ref_annotations))
    return selected_references


def add_reference_annotations(inputs: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    NOTE: this functionality is not robust, and quite frankly doesn't really work. Use at your own risk.

    Add reference annotations to the document dictionary. This function takes raw html as input. In other words, this
    function must be called before any transformation on the raw html occurs.

    Args:
        inputs: Dict with a 'content' key containing raw html.

    Returns: Dict with an additional 'references' key with list of string references.

    """
    for entity_id in inputs:
        potential_references = extract_potential_references(inputs[entity_id]['content'])
        annotated_potential_references = annotate_potential_references(potential_references)

        # TODO: switch to actually evaluating potential reference candidates
        # selected_references = evaluate_potential_references(annotated_potential_references)
        selected_references = annotated_potential_references

        inputs[entity_id]['references'] = selected_references
    return inputs
