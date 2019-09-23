import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# copied from
# https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python/46564234#46564234


class Splitter:
    """
    split the document into sentences and tokenize each sentence
    """

    def __init__(self):
        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        out : ['What', 'can', 'I', 'say', 'about', 'this', 'place', '.']
        """
        # split into single sentence
        sentences = self.splitter.tokenize(text)
        # tokenization in each sentences
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        return tokens


class LemmatizationWithPOSTagger:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

    def pos_tag(self, tokens):
        # find the pos tagginf for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
        pos_tokens = [nltk.pos_tag(token) for token in tokens]

        # lemmatization using pos tagg
        # convert into feature set of [('What', 'What', ['WP']), ('can', 'can', ['MD']), ...
        # ie [original WORD, Lemmatized word, POS tag]
        pos_tokens = [
            [(word, self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos_tag)), [pos_tag]) for (word, pos_tag) in
             pos] for pos in pos_tokens]
        return pos_tokens


class Lemmatizer:

    def __init__(self):
        self.splitter = Splitter()
        self.lemmatizer = LemmatizationWithPOSTagger()

    @staticmethod
    def reassemble(list_of_sent_list_of_tokens):
        return ' '.join([token[1] for sentence in list_of_sent_list_of_tokens for token in sentence])

    def lemmatize(self, text):
        tokens = self.splitter.split(text)
        lemma_pos_token = self.lemmatizer.pos_tag(tokens)
        return self.reassemble(lemma_pos_token)

    def lemmatize_list_of_texts(self, list_of_texts):
        lemmatized_list_of_texts = []
        for text in list_of_texts:
            lemmatized_list_of_texts.append(self.lemmatize(text))
        return lemmatized_list_of_texts
