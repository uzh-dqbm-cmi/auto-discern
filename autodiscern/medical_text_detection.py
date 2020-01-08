# determine whether the webpage is medical or not

# === EXAMPLE USAGE ===
# %load_ext autoreload
# %autoreload 2
# from autodiscern import medical_text_detection as mtd
# test_results = mtd.integration_test()
# test_results = mtd.integration_test('wikipedia kitten')
# mtd.view_relevant_metamap_details(test_results[0])
# === === === === ===

import requests
import autodiscern.transformations as adt
import autodiscern.annotations as ada


def determine_medical_yn_metamap(data_dict):
    """
    Determine whether the webpage is medical or not.

    data_dict  = {'content': html_page, 'url': url}
    """
    # prep the data dict into the right format for the transformer
    data_dict['id'] = 0
    data_dict_of_dicts = {0: data_dict}

    html_transformer = adt.Transformer(leave_some_html=False, parallelism=False)
    transformed_data = html_transformer.apply(data_dict_of_dicts)
    transformed_data = ada.add_inline_citations_annotations(transformed_data)
    transformed_data = ada.add_metamap_annotations(transformed_data)

    # get number of references to medical conditions
    metamap_references = transformed_data[0]['metamap']
    medical_disorder_ref_cnt = sum([1 for ref in metamap_references if ref == 'Disorders'])
    medical_drugs_ref_cnt = sum([1 for ref in metamap_references if ref == 'Chemicals & Drugs'])
    medical_procedures_ref_cnt = sum([1 for ref in metamap_references if ref == 'Procedures'])

    result = False
    if medical_disorder_ref_cnt > 3:
        if medical_drugs_ref_cnt + medical_procedures_ref_cnt > 3:
            result = True

    things_to_report = {
        'result': result,
        'metamap': transformed_data[0]['metamap'],
        'metamap_detail': transformed_data[0]['metamap_detail'],
        'medical_disorder_ref_cnt': medical_disorder_ref_cnt,
        'medical_drugs_ref_cnt': medical_drugs_ref_cnt,
        'medical_procedures_ref_cnt': medical_procedures_ref_cnt,
    }
    return things_to_report


def test_determine_medical_yn_metamap(url):
    res = requests.get(url)
    html_page = res.content.decode("utf-8")
    data_dict = {'content': html_page, 'url': url}
    return determine_medical_yn_metamap(data_dict)


def integration_test(subset=None):
    """
    Integration test for test_determine_medical_yn_metamap

    Args:
        subset: List[str]. optional way to select a subset of test cases, by their name

    Returns: Dict of test results

    """

    test_cases = [
        {
            'name': 'mayo clinic depression',
            'url': 'https://www.mayoclinic.org/diseases-conditions/depression/symptoms-causes/syc-20356007',
            'expected_result': True,
        },
        {
            'name': 'wikipedia kitten',
            'url': 'https://en.wikipedia.org/wiki/Kitten',
            'expected_result': False,
        },
        {
            'name': 'classification',
            'url': 'https://en.wikipedia.org/wiki/One-class_classification',
            'expected_result': False,
        }
    ]

    test_results = []
    for test_case in test_cases:
        if subset is None or test_case['name'] in subset:
            print("Running test case '{}'".format(test_case['name']))
            test_result = test_determine_medical_yn_metamap(test_case['url'])
            test_results.append(test_result)

            if test_result['result'] == test_case['expected_result']:
                print("=== Test case '{}' passed ===\n".format(test_case['name']))
            else:
                print(">>> Test case '{}' FAILED <<<\n".format(test_case['name']))

    return test_results


def view_relevant_metamap_details(test_result):
    for i, category in enumerate(test_result['metamap']):
        if category in ['Disorders', 'Chemicals & Drugs', 'Procedures']:
            print(category)
            print('trigger: {}'.format(test_result['metamap_detail'][i]['trigger']))
            print('preferred_name: {}'.format(test_result['metamap_detail'][i]['preferred_name']))
            print('semtypes: {}'.format(test_result['metamap_detail'][i]['semtypes']))
            print()
            # print(test_result['metamap_detail'][i])
