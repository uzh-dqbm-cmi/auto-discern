import unittest
import autodiscern.transformations as adt


class TestTransformations(unittest.TestCase):

    def test_remove_html(self):
        input = "<h1>I am a Header</h1>"
        expected_output = "I am a Header"
        self.assertEqual(adt.remove_html(input), expected_output)
