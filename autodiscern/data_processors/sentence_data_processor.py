from autodiscern import DataManager
import argparse


TAG = 'sentence_with_mm_and_ner_ammendments'


def sentence_data_preprocessor(data_dict):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from autodiscern import transformations, annotations, model

    html_to_sentence_transformer = transformations.Transformer(leave_some_html=True,
                                                               html_to_plain_text=True,
                                                               segment_into='sentences',
                                                               flatten=True,
                                                               remove_newlines=False,
                                                               annotate_html=True,
                                                               parallelism=False)
    transformed_data = html_to_sentence_transformer.apply(data_dict)

    # do links before punctuation removal because need punc for link id,
    # and before metamap because otherwise metamap will find references to medical terms in links
    transformed_data = annotations.add_inline_citations_annotations(transformed_data)
    transformed_data = annotations.amend_content_with_link_plain_text(transformed_data)

    # remove punctuation with bad encodings before metamap because it messes up metamap character indexing
    # (grr encoding!)
    transformed_data = annotations.ammed_content_replace_bad_punctuation_encoding(transformed_data)

    # do metamap before ner, becauase otherwise ner will replace portions of medical terms
    transformed_data = annotations.add_metamap_annotations(transformed_data)
    transformed_data = annotations.amend_content_with_metamap_concepts(transformed_data)

    transformed_data = annotations.add_ner_annotations(transformed_data)
    transformed_data = annotations.amend_content_with_ner_type_labels(transformed_data)

    sid = SentimentIntensityAnalyzer()

    for key in transformed_data:
        transformed_data[key]['feature_vec'] = model.build_remaining_feature_vector(transformed_data[key], sid)

    return transformed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store', type=int, default=3,
                        help='Run the data processor on a subset of data for faster testing.')
    args = parser.parse_args()

    discern_path = "~/switchdrive/Institution/discern"
    dm = DataManager(discern_path)
    raw_data_dict = dm.build_dicts()

    if args.test:
        print("***Running in test mode on {} documents***".format(args.test))
        subset = list(raw_data_dict.keys())[:args.test]
        raw_data_dict = {key: raw_data_dict[key] for key in raw_data_dict if key in subset}
        TAG = "{}_test".format(TAG)

    file_name = dm.cache_data_processor(raw_data_dict, sentence_data_preprocessor,
                                        tag=TAG, enforce_clean_git=False)
    print("--> Completed caching {}".format(file_name))

    # re-load the processing func and re-run it on an input
    sent_data_processor = dm.load_cached_data_processor(file_name)

    # test the data
    retreived_data = sent_data_processor.data

    # test the code
    retrieved_code = sent_data_processor.view_code()

    # test the processing func
    example_key = list(raw_data_dict.keys())[0]
    example_item = {example_key: raw_data_dict[example_key]}
    result = sent_data_processor.rerun(example_item)

    print("--> Re-load test complete")
