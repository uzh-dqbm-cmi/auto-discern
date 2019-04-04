from bs4 import BeautifulSoup
import unittest
import autodiscern.transformations as adt


class TestTransformations(unittest.TestCase):

    def test_replace_problem_chars(self):
        test_input = "words \twords\twords"
        expected_output = "words  words words"
        self.assertEqual(adt.Transformer.replace_chars(test_input, ['\t'], ' '), expected_output)

    def test_regex_out_punctuation_and_white_space_replaces_extra_consecutive_chars(self):
        test_input = "text text..\n. text"
        expected_output = "text text. \ntext"
        self.assertEqual(adt.Transformer.regex_out_punctuation_and_white_space(test_input), expected_output)

    def test_regex_out_punctuation_and_white_space_no_effect_single_period(self):
        test_input = "text."
        self.assertEqual(adt.Transformer.regex_out_punctuation_and_white_space(test_input), test_input)

    def test_regex_out_punctuation_and_white_space_removes_double_space_between_words(self):
        test_input = "text  text."
        expected_output = "text text."
        self.assertEqual(adt.Transformer.regex_out_punctuation_and_white_space(test_input), expected_output)

    def test_regex_out_punctuation_and_white_space_no_effect_period_between_words(self):
        test_input = "text. text"
        expected_output = "text. text"
        self.assertEqual(adt.Transformer.regex_out_punctuation_and_white_space(test_input), expected_output)

    def test_regex_out_punctuation_and_white_space_removes_extra_consecutive_periods(self):
        test_input = "text text..."
        expected_output = "text text. "
        self.assertEqual(adt.Transformer.regex_out_punctuation_and_white_space(test_input), expected_output)

    def test_regex_out_punctuation_and_white_space_removes_leading_period(self):
        test_input = "\n. \ntext text. "
        expected_output = "text text. "
        self.assertEqual(adt.Transformer.regex_out_punctuation_and_white_space(test_input), expected_output)

    def test_regex_out_punctuation_and_white_space_combines_question_mark_period(self):
        test_input = "text text?. "
        expected_output = "text text? "
        self.assertEqual(adt.Transformer.regex_out_punctuation_and_white_space(test_input), expected_output)

    def test_condense_line_breaks_multiple_newlines(self):
        test_input = "text\n\ntext"
        expected_output = "text \ntext"
        self.assertEqual(adt.Transformer.condense_line_breaks(test_input), expected_output)

    def test_condense_line_breaks_strips(self):
        test_input = "text\n"
        expected_output = "text"
        self.assertEqual(adt.Transformer.condense_line_breaks(test_input), expected_output)

    def test_condense_line_breaks_replaces_single_break_html_tag(self):
        test_input = "text<br>text"
        expected_output = "text\ntext"
        self.assertEqual(adt.Transformer.condense_line_breaks(test_input), expected_output)

    def test_condense_line_breaks_replaces_multiple_break_html_tags(self):
        test_input = "text<br><br>text"
        expected_output = "text \ntext"
        self.assertEqual(adt.Transformer.condense_line_breaks(test_input), expected_output)

    def test_condense_line_breaks_replaces_break_html_tags_with_bs4_slash(self):
        test_input = "text<br/>text"
        expected_output = "text\ntext"
        self.assertEqual(adt.Transformer.condense_line_breaks(test_input), expected_output)

    def test_condense_line_breaks_replaces_combo_break_html_tag_and_newline(self):
        test_input = "text<br>\ntext"
        expected_output = "text \ntext"
        self.assertEqual(adt.Transformer.condense_line_breaks(test_input), expected_output)

    def test_soup_to_text_with_tags_replaces_amp(self):
        test_input = BeautifulSoup('<html><body><h2 class="selectedHighlight">Staging, grading &amp; treatment</h2></body></html>', features="html.parser")
        expected_output = '<html><body><h2 class="selectedHighlight">Staging, grading & treatment</h2></body></html>'
        self.assertEqual(adt.Transformer.soup_to_text_with_tags(test_input), expected_output)

    def test_reformat_html_link_tags_replaces_link_with_domain(self):
        test_input = BeautifulSoup('<html><body>There is more information on medication on the <a title="Royal College of Psychiatrists information on medications" href="http://www.rcpsych.ac.uk/mentalhealthinformation/therapies.aspx ">website of the Royal College of Psychiatrists</a></body></html>', features="html.parser")
        expected_output = BeautifulSoup('<html><body>There is more information on medication on the <a href="rcpsych">website of the Royal College of Psychiatrists</a></body></html>', features="html.parser")
        self.assertEqual(adt.Transformer.reformat_html_link_tags(test_input), expected_output)

    def test_replace_html(self):
        test_input = BeautifulSoup('<html><body><h1 font="Blue">Heading1</h1><h2><i>Heading2</i></h2><a href="google.com">link here</a></p></body></html>', features="html.parser")
        expected_output = '<h1>Heading1</h1>thisisanh2tag Heading2.<a href="google.com">link here</a>'

        tags_to_keep = {'h1'}
        tags_to_keep_with_attr = {'a'}
        tags_to_replace_with_str = {
            'h2': ('thisisanh2tag ', '.'),
        }
        default_tag_replacement_str = ''
        transformer = adt.Transformer()
        test_output = transformer.replace_html(test_input, tags_to_keep, tags_to_keep_with_attr,
                                               tags_to_replace_with_str, default_tag_replacement_str,
                                               include_link_domains=True)
        self.assertEqual(test_output, expected_output)

    def test_replace_html_keeps_tag(self):
        test_input = BeautifulSoup('<html><body><h1>Heading1</h1></body></html>', features="html.parser")
        expected_output = '<h1>Heading1</h1>'

        tags_to_keep = {'h1'}
        tags_to_keep_with_attr = set()
        tags_to_replace_with_str = {}
        default_tag_replacement_str = ''
        transformer = adt.Transformer()
        test_output = transformer.replace_html(test_input, tags_to_keep, tags_to_keep_with_attr,
                                               tags_to_replace_with_str, default_tag_replacement_str,
                                               include_link_domains=True)
        self.assertEqual(test_output, expected_output)

    def test_replace_html_replaces_tag(self):
        test_input = BeautifulSoup('<html><body><h1>Heading1</h1></body></html>', features="html.parser")
        expected_output = 'thisisah1tag Heading1. '

        tags_to_keep = set()
        tags_to_keep_with_attr = set()
        tags_to_replace_with_str = {'h1': ('thisisah1tag ', '. ')}
        default_tag_replacement_str = ''
        transformer = adt.Transformer()
        test_output = transformer.replace_html(test_input, tags_to_keep, tags_to_keep_with_attr,
                                               tags_to_replace_with_str, default_tag_replacement_str,
                                               include_link_domains=True)
        self.assertEqual(test_output, expected_output)

    def test_replace_html_replaces_link_no_domain(self):
        test_input = BeautifulSoup('<html><body>There is a <a href="google.com">link here</a>.</body></html>', features="html.parser")
        expected_output = 'There is a thisisalinktag link here.'

        tags_to_keep = set()
        tags_to_keep_with_attr = set()
        tags_to_replace_with_str = {'a': ('thisisalinktag ', '')}
        default_tag_replacement_str = ''
        transformer = adt.Transformer()
        test_output = transformer.replace_html(test_input, tags_to_keep, tags_to_keep_with_attr,
                                               tags_to_replace_with_str, default_tag_replacement_str,
                                               include_link_domains=False)
        self.assertEqual(test_output, expected_output)

    def test_replace_html_replaces_link_with_domain(self):
        test_input = BeautifulSoup('<html><body>There is a <a href="google.com">link here</a>.</body></html>', features="html.parser")
        expected_output = 'There is a thisisalinktaggoogle link here.'

        tags_to_keep = set()
        tags_to_keep_with_attr = set()
        tags_to_replace_with_str = {'a': ('thisisalinktag ', '')}
        default_tag_replacement_str = ''
        transformer = adt.Transformer()
        test_output = transformer.replace_html(test_input, tags_to_keep, tags_to_keep_with_attr,
                                               tags_to_replace_with_str, default_tag_replacement_str,
                                               include_link_domains=True)
        self.assertEqual(test_output, expected_output)

    def test_flatten_text_dicts(self):
        test_input = [
            {'id': 0, 'content': ['word0', 'word1']},
            {'id': 1, 'content': ['word2', 'word3']},
        ]
        expected_output = [
            {'id': 0, 'sub_id': 0, 'content': 'word0'},
            {'id': 0, 'sub_id': 1, 'content': 'word1'},
            {'id': 1, 'sub_id': 0, 'content': 'word2'},
            {'id': 1, 'sub_id': 1, 'content': 'word3'},
        ]
        self.assertEqual(adt.Transformer._flatten_text_dicts(test_input), expected_output)

    def test_annotate_and_clean_html_finds_and_cleans_tag(self):
        test_input = {
            'id': 0,
            'content': 'thisisah1tag I am a Header.'
        }
        expected_output = {
            'id': 0,
            'content': 'I am a Header.',
            'html_tags': ['h1'],
            'domains': [],
        }
        self.assertEqual(adt.Transformer._annotate_and_clean_html(test_input), expected_output)

    def test_annotate_and_clean_html_for_links_with_domain(self):
        test_input = {
            'id': 0,
            'content': 'thisisalinktaggoogle I am a Header.'
        }
        expected_output = {
            'id': 0,
            'content': 'I am a Header.',
            'html_tags': ['a'],
            'domains': ['google'],
        }
        self.assertEqual(adt.Transformer._annotate_and_clean_html(test_input), expected_output)

    def test_annotate_and_clean_html_for_multiple_links_with_domain(self):
        test_input = {
            'id': 0,
            'content': 'thisisalinktaggoogle I am a thisisalinktagmaps Header.'
        }
        expected_output = {
            'id': 0,
            'content': 'I am a Header.',
            'html_tags': ['a'],
            'domains': ['google', 'maps'],
        }
        self.assertEqual(adt.Transformer._annotate_and_clean_html(test_input), expected_output)


class TestAcceptanceTransformation(unittest.TestCase):

    def setUp(self):
        self.test_input_1 = {
            'id': 0,
            'content': """
            <div class="field-item even" property="content:encoded"><div id="selectedWebpagePart" contenteditable="false"><div id="selectedWebpagePart" contenteditable="false"><div class="mainCol2Col selectedHighlight">
                   <div class="topleader">
                   <div class="vsp"> </div>
                    <div class="leader ad"><br></div></div><div class="mainContent">
                        
                        <div class="articleHtml">
            <div class="toolbar_ns" style="float:right;margin-top:-3px">
            <table><tbody></tbody></table></div>    
            <script>
            <!--//--><![CDATA[// ><!--
             function createToolbar() {	 
                if ('Antidepressants') {
                    var st=readCookie("SAVVYTOPICS");if (!st || st.indexOf("|Antidepressants|")==-1) {
                        var desc=st?"Click here to add <i>Antidepressants to your list of topics.":"<strong>Stay up-to-date on the health topics that interest you.<br /><br />Click here to sign in or sign up for HealthSavvy, and add <i>Antidepressants to your list of topics.";
                        addToolbarButton("HealthSavvy", "tb_hsicon tool_sp", "#",  savvyClick, "HealthSavvy","hs_savvy_favorite",desc);}
                }
                addToolbarButton( "Send this Page","tb_mail tool_sp", "#", function(event) {emailPage(event);return false;}, "Send Page",null, "<strong>Send Using Facebook or Email.<br /><br />Click here to send this page using Facebook or email. You may add a personal message to the email.");
                addToolbarButton( "Print","tb_print tool_sp", "#", function(event) {printPage(event);return false;}, "Print Article",null, "Click here to print this page."); 	   
             }
             createToolbar();  
            
            //--><!]]>
            </script><h1>Antidepressants</h1>
                        <div id="pageOneHeader"><div>
            <h3>Antidepressants are medications primarily used for treating depression.</h3></div></div></div></div><div>
            <a name="chapter_0" href="http://depression.emedtv.com/undefined" id="chapter_0"></a><h2>What Are Antidepressants?</h2></div>
                            <div>
            Antidepressants are medications used to treat <a href="http://depression.emedtv.com/depression/depression.html" onmouseout="hideDescription(event);" onmouseover="showDescription(event, '/depression/depression.html', 'Depression causes unnecessary suffering for both people who have the illness and their families.', 'Depression')">depression</a>. Some of these medications&nbsp;are blue.</div>
            <div>&nbsp;</div>
            <div><em>(Click <a title="Antidepressant Uses" href="http://depression.emedtv.com/antidepressants/antidepressant-uses.html" onmouseover="showDescription(event, '/antidepressants/antidepressant-uses.html', 'Besides depression treatment, antidepressants are also approved for other uses.', 'Antidepressant Uses')" onmouseout="hideDescription(event);">Antidepressant Uses</a> for more information on what&nbsp;they are used for, including possible <a href="http://drugs.emedtv.com/medicine/off-label.html" onmouseout="hideDescription(event);" onmouseover="showDescription(event, 'http://drugs.emedtv.com/medicine/off-label.html', 'This eMedTV page defines an off-label use as one where a physician prescribes a medication to treat a condition, even though the FDA has not approved the medicine for that specific use.', 'Off-Label')">off-label</a> uses.)</em></div>
            <div>&nbsp;</div>
            <div>
            <a name="chapter_1" href="http://depression.emedtv.com/undefined" id="chapter_1"></a><h2>Types of Antidepressants</h2></div>
                            <div>
            There are several types of antidepressants available to treat depression.</div>
            <div>&nbsp;</div>
            </div></div></div></div>
        """
        }

        # create starting dict for expected output. add content key in individual tests
        self.expected_output = {'id': 0}

    def test_html_to_text(self):
        transformer = adt.Transformer(leave_some_html=False)

        test_input = self.test_input_1
        self.expected_output['content'] = """Antidepressants. 
Antidepressants are medications primarily used for treating depression. 
What Are Antidepressants? 
Antidepressants are medications used to treat depression. Some of these medications are blue. 
(Click Antidepressant Uses for more information on what they are used for, including possible off-label uses.) 
Types of Antidepressants. 
There are several types of antidepressants available to treat depression."""

        output = transformer.apply([test_input])
        self.assertEqual(output, [self.expected_output])

    def test_html_to_limited_html(self):
        transformer = adt.Transformer(leave_some_html=True)

        test_input = self.test_input_1
        self.expected_output['content'] = """<h1>Antidepressants</h1> 
<h3>Antidepressants are medications primarily used for treating depression.</h3>
<a href="emedtv"></a><h2>What Are Antidepressants?</h2> 
Antidepressants are medications used to treat <a href="emedtv">depression</a>. Some of these medications are blue. 
(Click <a href="emedtv">Antidepressant Uses</a> for more information on what they are used for, including possible <a href="emedtv">off-label</a> uses.) 
<a href="emedtv"></a><h2>Types of Antidepressants</h2> 
There are several types of antidepressants available to treat depression."""
        output = transformer.apply([test_input])
        self.assertEqual(output, [self.expected_output])

    def test_html_to_limited_html_plain_text(self):
        transformer = adt.Transformer(leave_some_html=True, html_to_plain_text=True)

        test_input = self.test_input_1
        self.expected_output['content'] = """thisisah1tag Antidepressants. 
thisisah3tag Antidepressants are medications primarily used for treating depression. 
thisisalinktagemedtv thisisah2tag What Are Antidepressants? 
Antidepressants are medications used to treat thisisalinktagemedtv depression . Some of these medications are blue. 
(Click thisisalinktagemedtv Antidepressant Uses for more information on what they are used for, including possible thisisalinktagemedtv off-label uses.) 
thisisalinktagemedtv thisisah2tag Types of Antidepressants. 
There are several types of antidepressants available to treat depression."""
        output = transformer.apply([test_input])
        self.assertEqual(output, [self.expected_output])

    def test_html_to_text_to_words(self):
        transformer = adt.Transformer(leave_some_html=False, segment_into='words')

        test_input = self.test_input_1
        self.expected_output['content'] = [
            "Antidepressants", ".",
            "Antidepressants", "are", "medications", "primarily", "used", "for", "treating", "depression", ".",
            "What", "Are", "Antidepressants", "?",
            "Antidepressants", "are", "medications", "used", "to", "treat", "depression", ".",
            "Some", "of", "these", "medications", "are", "blue", ".",
            "(", "Click", "Antidepressant", "Uses", "for", "more", "information", "on", "what", "they", "are", "used",
            "for", ",", "including", "possible", "off", "-", "label", "uses", ".", ")",
            "Types", "of", "Antidepressants", ".",
            "There", "are", "several", "types", "of", "antidepressants", "available", "to", "treat", "depression", ".",
        ]

        output = transformer.apply([test_input])
        self.assertEqual(output, [self.expected_output])

    def test_html_to_text_to_sentences(self):
        transformer = adt.Transformer(leave_some_html=False, segment_into='sentences')

        test_input = self.test_input_1
        self.expected_output['content'] = [
            "Antidepressants.",
            "Antidepressants are medications primarily used for treating depression.",
            "What Are Antidepressants?",
            "Antidepressants are medications used to treat depression.",
            "Some of these medications are blue.",
            "(Click Antidepressant Uses for more information on what they are used for, including possible off-label uses.)",
            "Types of Antidepressants.",
            "There are several types of antidepressants available to treat depression.",
        ]

        output = transformer.apply([test_input])
        self.assertEqual(output, [self.expected_output])

    def test_html_to_text_to_paragraphs(self):
        transformer = adt.Transformer(leave_some_html=False, segment_into='paragraphs')

        test_input = self.test_input_1
        self.expected_output['content'] = [
            "Antidepressants. ",
            "Antidepressants are medications primarily used for treating depression. ",
            "What Are Antidepressants? ",
            "Antidepressants are medications used to treat depression. Some of these medications are blue. ",
            "(Click Antidepressant Uses for more information on what they are used for, including possible off-label uses.) ",
            "Types of Antidepressants. ",
            "There are several types of antidepressants available to treat depression.",
        ]

        output = transformer.apply([test_input])
        self.assertEqual(output, [self.expected_output])

    # to_limited_html_plain_text segmentation tests

    def test_html_to_limited_html_plain_text_to_sentences(self):
        transformer = adt.Transformer(leave_some_html=True, html_to_plain_text=True, segment_into='sentences')

        test_input = self.test_input_1
        self.expected_output['content'] = [
            "thisisah1tag Antidepressants.",
            "thisisah3tag Antidepressants are medications primarily used for treating depression.",
            "thisisalinktagemedtv thisisah2tag What Are Antidepressants?",
            "Antidepressants are medications used to treat thisisalinktagemedtv depression .",
            "Some of these medications are blue.",
            "(Click thisisalinktagemedtv Antidepressant Uses for more information on what they are used for, including possible thisisalinktagemedtv off-label uses.)",
            "thisisalinktagemedtv thisisah2tag Types of Antidepressants.",
            "There are several types of antidepressants available to treat depression.",
        ]

        output = transformer.apply([test_input])
        self.assertEqual(output, [self.expected_output])

    def test_html_to_limited_html_plain_text_to_paragraphs(self):
        transformer = adt.Transformer(leave_some_html=True, html_to_plain_text=True, segment_into='paragraphs')

        test_input = self.test_input_1
        self.expected_output['content'] = [
            "thisisah1tag Antidepressants. ",
            "thisisah3tag Antidepressants are medications primarily used for treating depression. ",
            "thisisalinktagemedtv thisisah2tag What Are Antidepressants? ",
            "Antidepressants are medications used to treat thisisalinktagemedtv depression . Some of these medications are blue. ",
            "(Click thisisalinktagemedtv Antidepressant Uses for more information on what they are used for, including possible thisisalinktagemedtv off-label uses.) ",
            "thisisalinktagemedtv thisisah2tag Types of Antidepressants. ",
            "There are several types of antidepressants available to treat depression.",
        ]

        output = transformer.apply([test_input])
        self.assertEqual(output, [self.expected_output])

    def test_html_to_limited_html_plain_text_to_sentences_flattened(self):
        transformer = adt.Transformer(leave_some_html=True, html_to_plain_text=True, segment_into='sentences',
                                      flatten=True, annotate_html=False)

        test_input = self.test_input_1
        expected_output = [
            {
                'id': 0,
                'sub_id': 0,
                'content': "Antidepressants.",
            },
            {
                'id': 0,
                'sub_id': 1,
                'content': "Antidepressants are medications primarily used for treating depression.",
            },
            {
                'id': 0,
                'sub_id': 2,
                'content': "What Are Antidepressants?",
            },
            {
                'id': 0,
                'sub_id': 3,
                'content': "Antidepressants are medications used to treat depression .",
            },
            {
                'id': 0,
                'sub_id': 4,
                'content': "Some of these medications are blue.",
            },
            {
                'id': 0,
                'sub_id': 5,
                'content': "(Click Antidepressant Uses for more information on what they are used for, including possible off-label uses.)",
            },
            {
                'id': 0,
                'sub_id': 6,
                'content': "Types of Antidepressants.",
            },
            {
                'id': 0,
                'sub_id': 7,
                'content': "There are several types of antidepressants available to treat depression.",
            },
        ]

        output = transformer.apply([test_input])
        for i in range(len(expected_output)):
            print(output[i])
            print(expected_output[i])
            self.assertDictEqual(output[i], expected_output[i])

    def test_html_to_limited_html_plain_text_to_sentences_flattened_annotated(self):
        transformer = adt.Transformer(leave_some_html=True, html_to_plain_text=True, segment_into='sentences',
                                      flatten=True, annotate_html=True)

        test_input = self.test_input_1
        expected_output = [
            {
                'id': 0,
                'sub_id': 0,
                'content': "Antidepressants.",
                'html_tags': ['h1'],
                'domains': [],
            },
            {
                'id': 0,
                'sub_id': 1,
                'content': "Antidepressants are medications primarily used for treating depression.",
                'html_tags': ['h3'],
                'domains': [],
            },
            {
                'id': 0,
                'sub_id': 2,
                'content': "What Are Antidepressants?",
                'html_tags': ['h2', 'a'],
                'domains': ['emedtv'],
            },
            {
                'id': 0,
                'sub_id': 3,
                'content': "Antidepressants are medications used to treat depression .",
                'html_tags': ['a'],
                'domains': ['emedtv'],
            },
            {
                'id': 0,
                'sub_id': 4,
                'content': "Some of these medications are blue.",
                'html_tags': [],
                'domains': [],
            },
            {
                'id': 0,
                'sub_id': 5,
                'content': "(Click Antidepressant Uses for more information on what they are used for, including possible off-label uses.)",
                'html_tags': ['a'],
                'domains': ['emedtv', 'emedtv'],
            },
            {
                'id': 0,
                'sub_id': 6,
                'content': "Types of Antidepressants.",
                'html_tags': ['h2', 'a'],
                'domains': ['emedtv'],
            },
            {
                'id': 0,
                'sub_id': 7,
                'content': "There are several types of antidepressants available to treat depression.",
                'html_tags': [],
                'domains': [],
            },
        ]

        output = transformer.apply([test_input])
        for i in range(len(expected_output)):
            print(output[i])
            print(expected_output[i])
            self.assertDictEqual(output[i], expected_output[i])
