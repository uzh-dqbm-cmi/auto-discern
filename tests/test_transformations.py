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

    def test_annotate_and_clean_html(self):
        test_input = {
            'id': 0,
            'content': 'thisisah1tag I am a Header.'
        }
        expected_output = {
            'id': 0,
            'content': 'I am a Header.',
            'html_tags': ['h1']
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
thisisalinktag thisisah2tag What Are Antidepressants? 
Antidepressants are medications used to treat thisisalinktag depression . Some of these medications are blue. 
(Click thisisalinktag Antidepressant Uses for more information on what they are used for, including possible thisisalinktag off-label uses.) 
thisisalinktag thisisah2tag Types of Antidepressants. 
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
            "thisisalinktag thisisah2tag What Are Antidepressants?",
            "Antidepressants are medications used to treat thisisalinktag depression .",
            "Some of these medications are blue.",
            "(Click thisisalinktag Antidepressant Uses for more information on what they are used for, including possible thisisalinktag off-label uses.)",
            "thisisalinktag thisisah2tag Types of Antidepressants.",
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
            "thisisalinktag thisisah2tag What Are Antidepressants? ",
            "Antidepressants are medications used to treat thisisalinktag depression . Some of these medications are blue. ",
            "(Click thisisalinktag Antidepressant Uses for more information on what they are used for, including possible thisisalinktag off-label uses.) ",
            "thisisalinktag thisisah2tag Types of Antidepressants. ",
            "There are several types of antidepressants available to treat depression.",
        ]

        output = transformer.apply([test_input])
        self.assertEqual(output, [self.expected_output])
