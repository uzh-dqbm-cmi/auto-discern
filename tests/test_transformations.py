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
        test_input = "words \twords\nwords"
        expected_output = "words  words words"
        self.assertEqual(adt.replace_problem_chars(test_input), expected_output)
