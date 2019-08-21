import pandas as pd
import unittest
import autodiscern.model as adm


class TestModel(unittest.TestCase):

    def test_vectorize_html(self, ):
        test_input = ['h1', 'a']
        expected_output = pd.DataFrame({
            'html_h1': 1,
            'html_h2': 0,
            'html_h3': 0,
            'html_h4': 0,
            'html_a':  1,
            'html_li': 0,
            'html_tr': 0,
        }, index=[0])
        pd.testing.assert_frame_equal(adm.vectorize_html(test_input), expected_output, check_like=True)

    def test_vectorize_metamap(self, ):
        test_input = ['Chemicals & Drugs', 'Chemicals & Drugs', 'Disorders', ]
        expected_output = pd.DataFrame({
            'MM-Chemicals & Drugs': 2,
            'MM-Disorders': 1,
            'MM-Activities & Behaviors': 0,
            'MM-Living Beings': 0,
            'MM-Genes & Molecular Sequences': 0,
            'MM-Anatomy': 0,
            'MM-Phenomena': 0,
            'MM-Occupations': 0,
            'MM-Physiology': 0,
            'MM-Concepts & Ideas': 0,
            'MM-Procedures': 0,
            'MM-Devices': 0,
            'MM-Objects': 0,
            'MM-Geographic Areas': 0,
            'MM-Organizations': 0,
        }, index=[0])
        pd.testing.assert_frame_equal(adm.vectorize_metamap(test_input), expected_output, check_like=True)

    def test_vectorize_link_type(self, ):
        test_input = ['internal', 'external', 'external']
        expected_output = pd.DataFrame({
            'internal_link_cnt': 1,
            'external_link_cnt': 2,
        }, index=[0])
        pd.testing.assert_frame_equal(adm.vectorize_link_type(test_input), expected_output, check_like=True)

    def test_vectorize_citations(self, ):
        test_input = ['[1]', '[2]']
        expected_output = pd.DataFrame({
            'inline_citation_cnt': 2,
        }, index=[0])
        pd.testing.assert_frame_equal(adm.vectorize_citations(test_input), expected_output, check_like=True)

    # def test_compute_polarity(self, ):
    #     test_input = 'sentence'
    #     expected_output = pd.DataFrame({
    #         'neg',
    #         'neu',
    #         'pos',
    #         'compound',
    #     }, index=[0])
    #     pd.testing.assert_frame_equal(adm.compute_polarity(test_input), expected_output, check_like=True)
