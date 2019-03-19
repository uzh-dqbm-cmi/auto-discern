import unittest
import autodiscern.transformations as adt


class TestTransformations(unittest.TestCase):

    def test_remove_html_removes_tags(self):
        test_input = "<h1>I am a Header</h1>"
        expected_output = "I am a Header"
        self.assertEqual(adt.remove_html(test_input), expected_output)

    def test_remove_selected_html_removes_some_keeps_others(self):
        test_input = "<div><h1>I am a Header</h1></div>"
        expected_output = "<h1>I am a Header</h1>"
        self.assertEqual(adt.remove_selected_html(test_input), expected_output)

    def test_replace_problem_chars(self):
        test_input = "words \twords\twords"
        expected_output = "words  words words"
        self.assertEqual(adt.replace_chars(test_input, ['\t'], ' '), expected_output)

    def test_regex_out_periods_and_white_space_replaces_extra_consecutive_chars(self):
        test_input = "text text..\n. text"
        expected_output = "text text. text"
        self.assertEqual(adt.regex_out_periods_and_white_space(test_input), expected_output)

    def test_regex_out_periods_and_white_space_no_effect_single_period(self):
        test_input = "text."
        self.assertEqual(adt.regex_out_periods_and_white_space(test_input), test_input)

    def test_regex_out_periods_and_white_space_removes_double_space_between_words(self):
        test_input = "text  text."
        expected_output = "text text."
        self.assertEqual(adt.regex_out_periods_and_white_space(test_input), expected_output)

    def test_regex_out_periods_and_white_space_no_effect_period_between_words(self):
        test_input = "text. text"
        expected_output = "text. text"
        self.assertEqual(adt.regex_out_periods_and_white_space(test_input), expected_output)

    def test_regex_out_periods_and_white_space_removes_extra_consecutive_periods(self):
        test_input = "text text..."
        expected_output = "text text. "
        self.assertEqual(adt.regex_out_periods_and_white_space(test_input), expected_output)

    def test_condense_line_breaks_multiple_newlines(self):
        test_input = "text\n\ntext"
        expected_output = "text\ntext"
        self.assertEqual(adt.condense_line_breaks(test_input), expected_output)

    def test_condense_line_breaks_strips(self):
        test_input = "text\n"
        expected_output = "text"
        self.assertEqual(adt.condense_line_breaks(test_input), expected_output)

    def test_condense_line_breaks_replaces_single_break_html_tag(self):
        test_input = "text<br>text"
        expected_output = "text\ntext"
        self.assertEqual(adt.condense_line_breaks(test_input), expected_output)

    def test_condense_line_breaks_replaces_multiple_break_html_tags(self):
        test_input = "text<br><br>text"
        expected_output = "text\ntext"
        self.assertEqual(adt.condense_line_breaks(test_input), expected_output)

    def test_condense_line_breaks_replaces_break_html_tags_with_bs4_slash(self):
        test_input = "text<br/>text"
        expected_output = "text\ntext"
        self.assertEqual(adt.condense_line_breaks(test_input), expected_output)

    def test_condense_line_breaks_replaces_combo_break_html_tag_and_newline(self):
        test_input = "text<br>\ntext"
        expected_output = "text\ntext"
        self.assertEqual(adt.condense_line_breaks(test_input), expected_output)