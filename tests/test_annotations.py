import unittest
import autodiscern.annotations as ada


class TestAnnotations(unittest.TestCase):

    def test_replace_metamap_with_concept_basic(self):
        test_input = {
            'content': '• Changes in appetite that result in weight losses or gains unrelated to dieting.',
            'metamap': ['Physiology', 'Disorders'],
            'metamap_detail': [
                {
                    'index': "'250-6'",
                    'mm': 'MMI',
                    'score': '5.59',
                    'preferred_name': 'Desire for food',
                    'cui': 'C0003618',
                    'semtypes': '[orgf]',
                    'trigger': '"Appetite"-text-0-"appetite"-NN-0',
                    'pos_info': '14/8',
                    'tree_codes': 'F02.830.071;G07.203.650.390.070;G10.261.390.070'
                },
                {
                    'index': "'250-6'",
                    'mm': 'MMI',
                    'score': '5.59',
                    'preferred_name': 'Weight decreased',
                    'cui': 'C1262477',
                    'semtypes': '[fndg]',
                    'trigger': '"Weight Losses"-text-0-"weight losses"-NN-0',
                    'pos_info': '38/13',
                    'tree_codes': 'C23.888.144.243.963;G07.345.249.314.120.200.963'
                }
            ],
        }
        output = ada.replace_metamap_content_with_concept_name(test_input['content'], test_input['metamap_detail'],
                                              test_input['metamap'])
        expected_output = '• Changes in MMConceptPhysiology that result in MMConceptDisorders or gains unrelated to dieting.'
        self.assertEqual(output, expected_output)

    def test_split_repeated_metamap_concepts(self):
        test_input = [{'index': '205',
                      'mm': 'MMI',
                      'score': '26.00',
                      'preferred_name': 'Brain',
                      'cui': 'C0006104',
                      'semtypes': '[bpoc]',
                      'trigger': '"Brain"-text-0-"brain"-NN-0',
                      'pos_info': '109/5;237/5',
                      'tree_codes': 'A08.186.211',
                      'concept': 'Anatomy'}]
        expected_ouput = [
            {
                'pos_info': '109/5',
                'start_pos': 109,
                'score': '26.00',
                'concept': 'Anatomy'
            },
            {
                'pos_info': '237/5',
                'start_pos': 237,
                'score': '26.00',
                'concept': 'Anatomy'
            }
        ]
        output = ada.split_repeated_metamap_concepts(test_input)
        for i in range(len(expected_ouput)):
            for key in expected_ouput[i]:
                self.assertEqual(output[i][key], expected_ouput[i][key])

    def test_prune_overlapping_metamap_details_no_overlap(self):
        test_input = [
            {'pos_info': '1/2', 'score': 1},
        ]
        expected_output = [
            {'pos_info': '1/2', 'score': 1},
        ]
        output = ada.prune_overlapping_metamap_details(test_input)
        self.assertEqual(output, expected_output)

    def test_prune_overlapping_metamap_details_overlap(self):
        test_input = [
            {'pos_info': '5/3', 'score': 1},
            {'pos_info': '6/2', 'score': .5},
        ]
        expected_output = [
            {'pos_info': '5/3', 'score': 1},
        ]
        output = ada.prune_overlapping_metamap_details(test_input)
        self.assertEqual(output, expected_output)

    def test_prune_overlapping_metamap_details_multiple_same(self):
        test_input = [
            {'pos_info': '5/3', 'score': 1},
            {'pos_info': '5/3', 'score': .5},
            {'pos_info': '5/3', 'score': .75},
        ]
        expected_output = [
            {'pos_info': '5/3', 'score': 1},
        ]
        output = ada.prune_overlapping_metamap_details(test_input)
        self.assertEqual(output, expected_output)

    def test_replace_metamap_content_with_concept_name_overlapping_concepts(self):
        test_input = {
            'content': '• Insomnia or oversleeping.',
            'metamap': ['Disorders', 'Disorders', 'Disorders'],
            'metamap_detail': [
                {
                    'index': "'250-7'",
                    'mm': 'MMI',
                    'score': '4.67',
                    'preferred_name': 'Insomnia, CTCAE 3.0',
                    'cui': 'C1963237',
                    'semtypes': '[fndg]',
                    'trigger': '"Insomnia"-text-0-"Insomnia"-NNP-0',
                    'pos_info': '3/8',
                    'tree_codes': 'C10.886.425.800.800;F03.870.400.800.800'
                },
                {
                    'index': "'250-7'",
                    'mm': 'MMI',
                    'score': '4.67',
                    'preferred_name': 'Insomnia, CTCAE 5.0',
                    'cui': 'C4554626',
                    'semtypes': '[fndg]',
                    'trigger': '"Insomnia"-text-0-"Insomnia"-NNP-0',
                    'pos_info': '3/8',
                    'tree_codes': 'C10.886.425.800.800;F03.870.400.800.800'
                },
                {
                    'index': "'250-7'",
                    'mm': 'MMI',
                    'score': '4.67',
                    'preferred_name': 'Sleeplessness',
                    'cui': 'C0917801',
                    'semtypes': '[sosy]',
                    'trigger': '"Insomnia"-text-0-"Insomnia"-NNP-0',
                    'pos_info': '3/8',
                    'tree_codes': 'C10.886.425.800.800;F03.870.400.800.800'
                }
            ],
        }
        output = ada.replace_metamap_content_with_concept_name(test_input['content'], test_input['metamap_detail'],
                                              test_input['metamap'])
        expected_output = '• MMConceptDisorders or oversleeping.'
        self.assertEqual(output, expected_output)

    def test_apply_inline_citation_regex_name_and_year_parens(self):
        test_input = "text (Frood, 1942)."
        expected_output = ["(Frood, 1942)"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_name_and_year_square_brackets(self):
        test_input = "text [Frood, 1942]."
        expected_output = ["[Frood, 1942]"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_name_et_al_and_year_parens(self):
        test_input = "text (Frood et al., 1942)."
        expected_output = ["(Frood et al., 1942)"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_multiple_parens(self):
        test_input = "text (Frood, 1942; Dent, 1944)."
        expected_output = ["(Frood, 1942; Dent, 1944)"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_two_distinct_citations(self):
        test_input = "text (Frood, 1942), (Dent, 1944)."
        expected_output = ["(Frood, 1942)", "(Dent, 1944)"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_citation_no_year_no_match(self):
        test_input = "text (Frood)."
        expected_output = []
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_citation_two_digit_year_no_match(self):
        test_input = "text (Frood, 98)."
        expected_output = []
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_citation_no_parens_no_match(self):
        test_input = "text from Frood, 1942."
        expected_output = []
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_lone_year_parens(self):
        test_input = "text (1942)."
        expected_output = ["(1942)"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_non_year_four_digit_num_parens_no_match(self):
        test_input = "text (1234)."
        expected_output = []
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_multi_numbers_no_match(self):
        test_input = "text (1234, 2019)."
        expected_output = []
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_lone_year_square_brackets(self):
        test_input = "text [1942]."
        expected_output = ["[1942]"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_single_digit_square_brackets(self):
        test_input = "text [1]."
        expected_output = ["[1]"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_single_digit_parens_no_match(self):
        test_input = "text (1)."
        expected_output = []
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_double_digit_square_brackets(self):
        test_input = "text [42]."
        expected_output = ["[42]"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_list_in_square_brackets(self):
        test_input = "text [1,2]."
        expected_output = ["[1,2]"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_list_with_space_in_square_brackets(self):
        test_input = "text [1, 2]."
        expected_output = ["[1, 2]"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_range_in_square_brackets(self):
        test_input = "text [1-3]."
        expected_output = ["[1-3]"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_range_in_square_brackets_with_spaces(self):
        test_input = "text [1 - 3]."
        expected_output = ["[1 - 3]"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_list_and_range_in_square_brackets(self):
        test_input = "text [1,3-5]."
        expected_output = ["[1,3-5]"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_list_and_range_with_space_in_square_brackets(self):
        test_input = "text [1, 3-5]."
        expected_output = ["[1, 3-5]"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_two_distinct_citations_different_types(self):
        test_input = "text [1] and text (Frood, 1942)."
        expected_output = ["[1]", "(Frood, 1942)"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_apply_inline_citation_regex_separate_parens_doesnt_get_roped_in(self):
        test_input = "(CCBT) ; NICE Technology Appraisal (2006)"
        expected_output = ["(2006)"]
        self.assertEqual(ada.apply_inline_citation_regex(test_input), expected_output)

    def test_extract_potential_references(self):
        example_input = """
        <p>Thank you, we just sent a survey email to confirm your preferences. </p>
        <h4 class=references__title>Further reading and references</h4>
        <i class="icon references__toggle"> 
            <svg role=presentation><use xlink:href=#chevron-down></use></svg>
        </i>
        <div class=references__content>
            <ul class="list references__list u-mb">
                <li><p><cite><a href=http://www.nice.org.uk/guidance/cg90/chapter/introduction target=_blank rel=noopener>Depression in adults: recognition and management</a></cite>; NICE Clinical Guideline (April 2016)</p></li>
                <li><p><cite><a href=http://cks.nice.org.uk/depression target=_blank rel=noopener>Depression</a></cite>; NICE CKS, October 2015 (UK access only)</p></li>
                <li><p><cite><a href="http://www.ncbi.nlm.nih.gov/entrez/query.fcgi?cmd=Retrieve&amp;db=PubMed&amp;dopt=Abstract&amp;list_uids=22786489" target=_blank rel=noopener>Rimer J, Dwan K, Lawlor DA, et al</a></cite>; Exercise for depression. Cochrane Database Syst Rev. 2012 Jul 117:CD004366.</p></li>
                <li><p><cite><a href="http://www.ncbi.nlm.nih.gov/entrez/query.fcgi?cmd=Retrieve&amp;db=PubMed&amp;dopt=Abstract&amp;list_uids=22674921" target=_blank rel=noopener>Chalder M, Wiles NJ, Campbell J, et al</a></cite>; Facilitated physical activity as a treatment for depressed adults: randomised BMJ. 2012 Jun 6344:e2758. doi: 10.1136/bmj.e2758.</p></li>
            </ul>
        </div>
        """
        expected_output = [
            "Depression in adults: recognition and management; NICE Clinical Guideline (April 2016)",
            "Depression; NICE CKS, October 2015 (UK access only)",
            "Rimer J, Dwan K, Lawlor DA, et al; Exercise for depression. Cochrane Database Syst Rev. 2012 Jul 117:CD004366.",
            "Chalder M, Wiles NJ, Campbell J, et al; Facilitated physical activity as a treatment for depressed adults: randomised BMJ. 2012 Jun 6344:e2758. doi: 10.1136/bmj.e2758.",
        ]
        output = ada.extract_potential_references(example_input)
        self.assertEqual(output, expected_output)

    def test_evaluate_potential_references(self):
        example_input = [
            ('This is not a citation.',
             [
                 ('This', 'note'),
                 ('is', 'note'),
                 ('not', 'note'),
                 ('a', 'note'),
                 ('citation.', 'note'),
             ]),
            ("You can do this online at www.mhra.gov.uk/yellowcard.",
             [
                 ('You', 'title'),
                 ('can', 'title'),
                 ('do', 'title'),
                 ('this', 'note'),
                 ('online', 'note'),
                 ('at', 'note'),
                 ('www.mhra.gov.uk/yellowcard.', 'note'),
             ]),
            ("The national guideline published in 2009 by the National Institute for Health and Care Excellence (NICE) and updated in 2016 advises regular exercise as a possible treatment.",
             [
                 ('The',
                'title'),
                 ('national', 'title'),
                 ('guideline', 'title'),
                 ('published', 'title'),
                 ('in', 'title'),
                 ('2009', 'date'),
                 ('by', 'note'),
                 ('the', 'note'),
                 ('National', 'institution'),
                 ('Institute', 'institution'),
                 ('for', 'institution'),
                 ('Health', 'institution'),
                 ('and', 'institution'),
                 ('Care', 'title'),
                 ('Excellence', 'title'),
                 ('(NI )',  'title'),
                 ('and', 'title'),
                 ('updated', 'title'),
                 ('in', 'title'),
                 ('2016', 'date'),
                 ('advises', 'title'),
                 ('regular', 'title'),
                 ('exercise', 'title'),
                 ('as', 'title'),
                 ('a', 'title'),
                 ('possible', 'title'),
                 ('treatment.', 'title'),
             ]),
            ("Chalder M, Wiles NJ, Campbell J, et al; Facilitated physical activity as a treatment for depressed adults: randomised BMJ. 2012 Jun 6344:e2758. doi: 10.1136/bmj.e2758.",
             [
                 ('Chalder', 'author'),
                 ('M,', 'author'),
                 ('Wiles', 'author'),
                 ('NJ,', 'author'),
                 ('Campbell', 'author'),
                 ('J,', 'author'),
                 ('et', 'author'),
                 ('al;', 'author'),
                 ('Facilitated', 'title'),
                 ('physical', 'title'),
                 ('activity', 'title'),
                 ('as', 'title'),
                 ('a', 'title'),
                 ('treatment', 'title'),
                 ('for', 'title'),
                 ('depressed', 'title'),
                 ('adults:', 'title'),
                 ('randomised', 'title'),
                 ('BMJ.', 'title'),
                 ('2012', 'date'),
                 ('Jun', 'date'),
                 ('6344:e2758.', 'date'),
                 ('doi:', 'date'),
                 ('10.1136/bmj.e2758.', 'pages'),
             ]),
        ]
        expected_output = [
            ("Chalder M, Wiles NJ, Campbell J, et al; Facilitated physical activity as a treatment for depressed adults: randomised BMJ. 2012 Jun 6344:e2758. doi: 10.1136/bmj.e2758.",
             [
                 ('Chalder', 'author'),
                 ('M,', 'author'),
                 ('Wiles', 'author'),
                 ('NJ,', 'author'),
                 ('Campbell', 'author'),
                 ('J,', 'author'),
                 ('et', 'author'),
                 ('al;', 'author'),
                 ('Facilitated', 'title'),
                 ('physical', 'title'),
                 ('activity', 'title'),
                 ('as', 'title'),
                 ('a', 'title'),
                 ('treatment', 'title'),
                 ('for', 'title'),
                 ('depressed', 'title'),
                 ('adults:', 'title'),
                 ('randomised', 'title'),
                 ('BMJ.', 'title'),
                 ('2012', 'date'),
                 ('Jun', 'date'),
                 ('6344:e2758.', 'date'),
                 ('doi:', 'date'),
                 ('10.1136/bmj.e2758.', 'pages'),
             ]),
        ]
        self.assertEqual(ada.evaluate_potential_references(example_input), expected_output)
