import unittest
import autodiscern.annotations as ada


class TestAnnotations(unittest.TestCase):

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
