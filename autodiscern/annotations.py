from bs4 import BeautifulSoup
from bs4.element import Tag
import re
from typing import Callable, Dict, List, Tuple
from autodiscern.data_manager import DataManager


def add_word_token_annotations(inputs: Dict[str, Dict]) -> Dict[str, Dict]:
    from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
    tok = WordTokenizer()
    for id in inputs:
        # have to convert tokens to text because spacy tokens are not pickleable
        inputs[id]['tokens'] = [t.text for t in tok.tokenize(inputs[id]['content'])]
    return inputs


def add_metamap_annotations(inputs: Dict[str, Dict], dm: DataManager, metamap_path: str = None,
                            git_bash_pth: str = None) -> Dict[str, Dict]:
    from pymetamap import MetaMapLite

    if metamap_path is None:
        print("NOTE: no Metamap path provided. Using Laura's default")
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
    print("Extracing MetaMap concepts...")
    mm = MetaMapLite.get_instance(metamap_path, git_bash_pth=git_bash_pth)
    concepts, error = mm.extract_concepts(sentences, ids)

    print("Attaching Metamap concepts...")
    # add concepts with score above 1 and matching the semantics filter to input sentences
    for concept in concepts:
        concept_dict = {}
        if float(concept.score) > 1:
            semtypes = concept.semtypes.replace('[', '').replace(']', '').split(',')
            for semtype in semtypes:
                if semtype in semantic_type_filter:
                    for fld in concept._fields:
                        concept_dict[fld] = getattr(concept, fld)

                    # attach concept to input_dict
                    id = concept_dict['index']
                    id = id.replace('"', '').replace("'", '')
                    if id not in inputs.keys():
                        if int(id) in inputs.keys():
                            id = int(id)
                        else:
                            raise ValueError("ERROR: MetaMap index {} not found in input keys")

                    if 'metamap' not in inputs[id]:
                        inputs[id]['metamap'] = []
                        inputs[id]['metamap_detail'] = []
                    metamap_category = metamap_semantics[metamap_semantics['abbreviation'] == semtype
                                                         ]['group_name'].iloc[0]
                    inputs[id]['metamap'].append(metamap_category)
                    inputs[id]['metamap_detail'].append(concept_dict)
                    break

    print("Done annotating MetaMap concepts")
    return inputs


def amend_content_with_metamap_concepts(inputs: Dict[str, Dict]) -> Dict[str, Dict]:
    for id in inputs:
        if 'metamap' in inputs[id].keys():
            inputs[id]['content'] = replace_metamap_content_with_concept_name(inputs[id]['content'],
                                                                              inputs[id]['metamap_detail'],
                                                                              inputs[id]['metamap'],)
    return inputs


def get_metamap_pos(pos_info: str) -> Tuple[int, int]:
    pos_start, pos_end = pos_info.split('/')
    pos_start = int(pos_start)
    pos_end = pos_start + int(pos_end)
    return pos_start, pos_end


def replace_metamap_content_with_concept_name(content: str, metamap_detail: List[Dict], metamap_concepts: List[str]
                                              ) -> str:
    # add concepts directly into detail dicts
    for i, mm_d in enumerate(metamap_detail):
        metamap_detail[i]['concept'] = metamap_concepts[i]

    metamap_detail = split_repeated_metamap_concepts(metamap_detail)
    metamap_detail = sorted(metamap_detail, key=lambda k: k['start_pos'])
    metamap_detail = prune_overlapping_metamap_details(metamap_detail)

    # metamap position indexes are based on raw strings, where '\n' counts as two characters, but they are only counted
    #    as 1 by Python, which breaks the positon-based replacement. Convert newlines to 2-character placeholders.
    content = content.replace('\n', '^^')

    for mm_d in reversed(metamap_detail):
        pos_start, pos_end = get_metamap_pos(mm_d['pos_info'])
        concept = "MMConcept" + mm_d['concept']
        content = ''.join((content[:pos_start - 1], concept, content[pos_end - 1:]))

    # flip the new line conversion back
    content = content.replace('^^', '\n')
    return content


def split_repeated_metamap_concepts(metamap_details: List[Dict]) -> List[Dict]:
    metamap_details_split = []
    for metamap_entry in metamap_details:
        pos_entries = metamap_entry['pos_info'].split(';')
        for pos in pos_entries:
            metamap_details_split.append({
                'pos_info': pos,
                'start_pos': get_metamap_pos((pos))[0],
                'concept': metamap_entry['concept'],
                'score': metamap_entry['score'],
            })
    return metamap_details_split


def prune_overlapping_metamap_details(mm_d: List[Dict]) -> List[Dict]:
    """Iterate over the metamap details, removing adjacent concepts that overlap, until no overlaps are found.
    Assumes metamap concepts are listed in position order."""
    # set overlap_found to True to enter the loop for the first time
    # subsequently, overlap_found is assumed False until an overlap is found,
    #   at which point the for loop breaks and starts over from the beginning
    #   because removing items from the list resets the indexes
    overlap_found = True
    while overlap_found:
        overlap_found = False
        for i, d in enumerate(mm_d):
            if i <= len(mm_d) - 2:
                pos_start, pos_end = get_metamap_pos(d['pos_info'])
                next_pos_start, next_pos_end = get_metamap_pos(mm_d[i + 1]['pos_info'])
                if pos_start <= next_pos_start <= pos_end or pos_start <= next_pos_end <= pos_end:
                    overlap_found = True
                    # figure out which of the two concepts to remove
                    if mm_d[i]['score'] >= mm_d[i + 1]['score']:
                        del mm_d[i + 1]
                    else:
                        del mm_d[i]
                    break
    return mm_d


def add_ner_annotations(inputs: Dict[str, Dict]) -> Dict[str, Dict]:
    # is there a batch predictor?
    # https://allenai.github.io/allennlp-docs/api/allennlp.predictors.html#sentence-tagger
    from allennlp.predictors.predictor import Predictor

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


def add_inline_citations_annotations(inputs: Dict[str, Dict]) -> Dict[str, Dict]:
    for id in inputs:
        sentence = inputs[id]['content']
        in_line_citations = apply_inline_citation_regex(sentence)
        inputs[id]['citations'] = in_line_citations
    return inputs


def apply_inline_citation_regex(text):
    year_num = r"(?:19|20)[0-9][0-9]"
    author_with_year = r"[^()\d]*" + year_num
    multiple_authors = author_with_year + r"(?:;\s*" + author_with_year + r")*"
    author_either_bracket_type = r"\(" + multiple_authors + r"\)|\[" + multiple_authors + r"\]"
    citation_number_in_square_brackets = r"\[\d+(?:[-,\s]+\d+)*\]"
    regex = author_either_bracket_type + r"|" + citation_number_in_square_brackets

    matches = re.findall(regex, text)
    return matches


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
