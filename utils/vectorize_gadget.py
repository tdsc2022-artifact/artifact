import re
from gensim.models import Word2Vec
import numpy
from utils.vocabulary import UNK, PAD, Vocabulary_token
from typing import Dict, List, Counter
import warnings
warnings.filterwarnings("ignore")

# Sets for operators
operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--', '!~', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||',
    '+=', '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.', '+', '-', '*', '&', '/', '%', '<', '>', '^', '|',
    '=', ',', '?', ':', ';', '{', '}'
}
"""
Functionality to train Word2Vec model and vectorize gadgets
Buffers list of tokenized gadgets in memory
Trains Word2Vec model using list of tokenized gadgets
Uses trained model embeddings to create 2D gadget vectors
"""


class GadgetVectorizer:
    def __init__(self, config):
        self.gadgets = []
        self.config = config
        self.vector_length = config.hyper_parameters.vector_length
        self.forward_slices = 0
        self.backward_slices = 0
        self.tk_counter = Counter()

    """
    Takes a line of C++ code (string) as input
    Tokenizes C++ code (breaks down into identifier, variables, keywords, operators)
    Returns a list of tokens, preserving order in which they appear
    """

    @staticmethod
    def tokenize(line):
        tmp, w = [], []
        i = 0
        while i < len(line):
            # Ignore spaces and combine previously collected chars to form words
            if line[i] == ' ':
                tmp.append(''.join(w))
                tmp.append(line[i])
                w = []
                i += 1
            # Check operators and append to final list
            elif line[i:i + 3] in operators3:
                tmp.append(''.join(w))
                tmp.append(line[i:i + 3])
                w = []
                i += 3
            elif line[i:i + 2] in operators2:
                tmp.append(''.join(w))
                tmp.append(line[i:i + 2])
                w = []
                i += 2
            elif line[i] in operators1:
                tmp.append(''.join(w))
                tmp.append(line[i])
                w = []
                i += 1
            # Character appended to word list
            else:
                w.append(line[i])
                i += 1
        if (len(w) != 0):
            tmp.append(''.join(w))
            w = []
        # Filter out irrelevant strings
        res = list(filter(lambda c: c != '', tmp))
        return list(filter(lambda c: c != ' ', res))

    """
    Tokenize entire gadget
    Tokenize each line and concatenate to one long list
    """

    @staticmethod
    def tokenize_gadget(gadget):
        tokenized = []
        function_regex = re.compile('FUN(\d)+')
        backwards_slice = False
        for line in gadget:
            tokens = GadgetVectorizer.tokenize(line)
            tokenized += tokens
            if len(list(filter(function_regex.match, tokens))) > 0:
                backwards_slice = True
            else:
                backwards_slice = False
        return tokenized, backwards_slice

    @staticmethod
    def extract_attn(gadget, api_path):
        '''
        extract code attentions following:
            1. variable definition;
            2. control statement;
            3. api function call;

        :param gadget:
        :return
        '''
        attn = []
        with open(api_path, "r") as f:
            apis = f.read().split(',')
        for line in gadget:
            tokens = GadgetVectorizer.tokenize(line)
            set_tokens = set(tokens)
            if GadgetVectorizer.is_control(
                    set_tokens) or GadgetVectorizer.is_api_call(
                        set_tokens, apis) or GadgetVectorizer.is_var_def(line):
                attn.append(line)
        return attn

    @staticmethod
    def is_api_call(tokens, apis):
        '''
        judge if a code statement is a api statement
        '''
        for api in apis:
            if api in tokens:
                return True
        return False

    @staticmethod
    def is_var_def(line):
        '''
        TODO: regular matching
        judge if a code statement is a var definition statement
        '''
        is_var = re.compile(
            r'\b(?:(?:auto\s*|const\s*|unsigned\s*|signed\s*|register\s*|volatile\s*|static\s*|void\s*|short\s*|long\s*|char\s*|int\s*|float\s*|double\s*|_Bool\s*|complex\s*)+)(?:\s+\*?\*?\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*[\[;,=)]'
        )
        return is_var.match(line) is not None
        # return len(tokens) > 2 and (tokens[1][:3] == "VAR" or
        #                             (tokens[1] == "*"
        #                              and tokens[2][:3] == "VAR"))

    @staticmethod
    def is_control(tokens):
        '''
        judge if a code statement is a control statement
        '''
        return "if" in tokens or "else" in tokens or "switch" in tokens or "case" in tokens or "for" in tokens or "while" in tokens

    """
    Add input gadget to model
    Tokenize gadget and buffer it to list
    """

    def add_gadget(self, gadget):
        tokenized_gadget, backwards_slice = GadgetVectorizer.tokenize_gadget(
            gadget)
        self.gadgets.append(tokenized_gadget)
        self.tk_counter.update(tokenized_gadget)
        if backwards_slice:
            self.backward_slices += 1
        else:
            self.forward_slices += 1

    """
    Uses Word2Vec to create a vector for each gadget
    Gets a vector for the gadget by combining token embeddings
    Number of tokens used is min of number_of_tokens and 50
    """

    def vectorize(self, gadget):
        tokenized_gadget, backwards_slice = GadgetVectorizer.tokenize_gadget(
            gadget)
        vectors = numpy.zeros(shape=(50, self.vector_length))
        masks = numpy.zeros(shape=(50))
        _negative_value = -1e9  # mask value
        if backwards_slice:
            for i in range(min(len(tokenized_gadget), 50)):
                vectors[50 - 1 - i] = self.embeddings[tokenized_gadget[
                    len(tokenized_gadget) - 1 - i]]
            if (len(tokenized_gadget) < 50):
                for i in range(50 - len(tokenized_gadget)):
                    masks[i] = _negative_value

        else:
            for i in range(min(len(tokenized_gadget), 50)):
                vectors[i] = self.embeddings[tokenized_gadget[i]]
            if (len(tokenized_gadget) < 50):
                for i in range(50 - len(tokenized_gadget)):
                    masks[50 - 1 - i] = _negative_value
        return vectors, masks

    def vectorize2(self, gadget):
        tokenized_gadget, backwards_slice = GadgetVectorizer.tokenize_gadget(
            gadget)
        vectors = numpy.zeros(shape=(len(tokenized_gadget),
                                     self.vector_length))
        for i in range(len(tokenized_gadget)):
            vectors[i] = self.embeddings[tokenized_gadget[i]]
        return vectors, backwards_slice

    def local_vectorize(self, gadget, api_path):
        attn = GadgetVectorizer.extract_attn(gadget, api_path)
        if len(attn) == 0:
            attn.append(gadget[-1])
        tokenized_attn, _ = GadgetVectorizer.tokenize_gadget(attn)
        vectors = numpy.zeros(shape=(len(tokenized_attn), self.vector_length))
        for i in range(len(tokenized_attn)):
            vectors[i] = self.embeddings[tokenized_attn[i]]
        return vectors

    def _counter_to_dict(
            self,
            values: Counter,
            n_most_common: int = None,
            additional_values: List[str] = None) -> Dict[str, int]:
        dict_values = []
        if additional_values is not None:
            dict_values += additional_values
        dict_values += list(zip(*values.most_common(n_most_common)))[0]
        return {value: i for i, value in enumerate(dict_values)}

    def build_vocab(self, vocab_path):
        '''

        :param vocab_path:
        '''
        additional_tokens = [PAD, UNK]
        token_to_id = self._counter_to_dict(
            self.tk_counter, self.config.dataset.token.vocabulary_size,
            additional_tokens)
        vocab = Vocabulary_token(token_to_id)
        vocab.dump_vocabulary(vocab_path)

    """
    Done adding gadgets, now train Word2Vec model
    Only keep list of embeddings, delete model and list of gadgets
    """

    def train_model(self, w2v_path):
        # Set min_count to 1 to prevent out-of-vocabulary errors
        model = Word2Vec(self.gadgets,
                         min_count=1,
                         size=self.vector_length,
                         sg=1)
        self.embeddings = model.wv
        model.save(w2v_path)
        del model
        del self.gadgets
        
    def load_model(self, w2v_path):
        model = Word2Vec.load(w2v_path)
        self.embeddings = model.wv
