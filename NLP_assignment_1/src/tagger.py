"""
taggers
"""
from nltk import ConditionalFreqDist
from nltk.tag.util import untag
from itertools import izip
import re
from nltk import UnigramTagger

# **************** CONSTANTS *****************
REGEX_RULES = (
    [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
     (r'(The|the|A|a|An|an)$', 'AT'),  # articles
     (r'.*able$', 'JJ'),  # adjectives
     (r'.*ness$', 'NN'),  # nouns formed from adjectives
     (r'.*ly$', 'RB'),  # adverbs
     (r'.*s$', 'NNS'),  # plural nouns
     (r'.*ing$', 'VBG'),  # gerunds
     (r'.*ed$', 'VBD'),  # past tense verbs
     (r'.*', 'NN')  # nouns (default)
    ])


def accuracy(reference, test):
    if len(reference) != len(test):
        raise ValueError("Lists must have the same length.")
    num_correct = 0
    for x, y in izip(reference, test):
        if x == y:
            num_correct += 1
    return float(num_correct) / len(reference)


class TaggerI(object):
    def tag(self, tokens):
        raise NotImplementedError()

    def batch_tag(self, sentences):
        return [self.tag(sent) for sent in sentences]

    def evaluate(self, gold):
        tagged_sents = self.batch_tag([untag(sent) for sent in gold])
        gold_tokens = sum(gold, [])
        test_tokens = sum(tagged_sents, [])
        return accuracy(gold_tokens, test_tokens)




# ______________________________________________________


class SequentialBackoffTagger(TaggerI):
    """
    An abstract base class for taggers that tags words sequentially,
    left to right.  Tagging of individual words is performed by the
    method L{choose_tag()}, which should be defined by subclasses.  If
    a tagger is unable to determine a tag for the specified token,
    then its backoff tagger is consulted.
    """

    def __init__(self, backoff=None):
        if backoff is None:
            self._taggers = [self]
        else:
            self._taggers = [self] + backoff._taggers

    def _get_backoff(self):
        if len(self._taggers) < 2:
            return None
        else:
            return self._taggers[1]

    backoff = property(_get_backoff, doc='''The backoff tagger for this tagger.''')

    # backoff = property(_get_backoff, doc='''The backoff tagger for this tagger.''')

    def tag(self, tokens):
        # docs inherited from TaggerI
        tags = []
        for i in range(len(tokens)):
            tags.append(self.tag_one(tokens, i, tags))
        return zip(tokens, tags)

    def tag_one(self, tokens, index, history):
        """
        Determine an appropriate tag for the specified token, and
        return that tag.  If this tagger is unable to determine a tag
        for the specified token, then its backoff tagger is consulted.
        """
        tag = None
        for tagger in self._taggers:
            tag = tagger.choose_tag(tokens, index, history)
            if tag is not None:
                break
        return tag

    def choose_tag(self, tokens, index, history):
        """
        Decide which tag should be used for the specified token, and
        return that tag.  If this tagger is unable to determine a tag
        for the specified token, return C{None} -- do I{not} consult
        the backoff tagger.  This method should be overridden by
        subclasses of C{SequentialBackoffTagger}.
        """
        raise AssertionError('SequentialBackoffTagger is an abstract class')


class DefaultTagger(SequentialBackoffTagger):
    """
    A tagger that assigns the same tag to every token.
    """

    def __init__(self, tag):
        """
        Construct a new tagger that assigns C{tag} to all tokens.
        """
        self._tag = tag
        SequentialBackoffTagger.__init__(self, None)

    def choose_tag(self, tokens, index, history):
        return self._tag  # ignore token and history

    def __repr__(self):
        return '<DefaultTagger: tag=%s>' % self._tag


class RegexpTagger(SequentialBackoffTagger):
    """
    A tagger that assigns tags to words based on regular expressions over word strings.
    """

    def __init__(self, regexps, backoff=None):
        self._regexps = regexps
        SequentialBackoffTagger.__init__(self, backoff)

    def choose_tag(self, tokens, index, history):
        for regexp, tag in self._regexps:
            # ignore history
            if re.match(regexp, tokens[index]):
                return tag
        return None

    def __repr__(self):
        return '<Regexp Tagger: size=%d>' % len(self._regexps)


class ContextTagger(SequentialBackoffTagger):
    """
    An abstract base class for sequential backoff taggers that choose
    a tag for a token based on the value of its "context".  Different
    subclasses are used to define different contexts.

    A C{ContextTagger} chooses the tag for a token by calculating the
    token's context, and looking up the corresponding tag in a table.
    This table can be constructed manually; or it can be automatically
    constructed based on a training corpus, using the L{_train()}
    factory method.

    @ivar _context_to_tag: Dictionary mapping contexts to tags.
    """

    def __init__(self, context_to_tag, backoff=None):
        SequentialBackoffTagger.__init__(self, backoff)
        if context_to_tag:
            self._context_to_tag = context_to_tag
        else:
            self._context_to_tag = {}

    def context(self, tokens, index, history):
        """
        @return: the context that should be used to look up the tag
            for the specified token; or C{None} if the specified token
            should not be handled by this tagger.
        @rtype: (hashable)
        """
        raise AssertionError('Abstract base class')

    def choose_tag(self, tokens, index, history):
        context = self.context(tokens, index, history)
        return self._context_to_tag.get(context)

    def _train(self, tagged_corpus, cutoff=0, verbose=False):
        """
        Initialize this C{ContextTagger}'s L{_context_to_tag} table
        based on the given training data.  In particular, for each
        context C{I{c}} in the training data, set
        C{_context_to_tag[I{c}]} to the most frequent tag for that
        context.  However, exclude any contexts that are already
        tagged perfectly by the backoff tagger(s).

        @param tagged_corpus: A tagged corpus.  Each item should be
            a C{list} of C{(word, tag)} tuples.
        @param cutoff: If the most likely tag for a context occurs
            fewer than C{cutoff} times, then exclude it from the
            context-to-tag table for the new tagger.
        """

        token_count = hit_count = 0

        # A context is considered 'useful' if it's not already tagged
        # perfectly by the backoff tagger.
        useful_contexts = set()

        # Count how many times each tag occurs in each context.
        fd = ConditionalFreqDist()
        for sentence in tagged_corpus:
            tokens, tags = zip(*sentence)
            for index, (token, tag) in enumerate(sentence):
                # Record the event.
                token_count += 1
                context = self.context(tokens, index, tags[:index])
                if context is None:
                    continue
                fd[context].inc(tag)
                # If the backoff got it wrong, this context is useful:
                if (self.backoff is None or
                            tag != self.backoff.tag_one(tokens, index, tags[:index])):
                    useful_contexts.add(context)

        # Build the context_to_tag table -- for each context, figure
        # out what the most likely tag is.  Only include contexts that
        # we've seen at least `cutoff` times.
        for context in useful_contexts:
            best_tag = fd[context].max()
            hits = fd[context][best_tag]
            if hits > cutoff:
                self._context_to_tag[context] = best_tag
                hit_count += hits


class NgramTagger(ContextTagger):
    """
    A tagger that chooses a token's tag based on its word string and
    on the preceeding I{n} word's tags.  In particular, a tuple
    C{(tags[i-n:i-1], words[i])} is looked up in a table, and the
    corresponding tag is returned.  N-gram taggers are typically
    trained on a tagged corpus.
    """

    def __init__(self, n, train=None, model=None,
                 backoff=None, cutoff=0, verbose=False):
        self._n = n
        ContextTagger.__init__(self, model, backoff)
        if train:
            self._train(train, cutoff, verbose)

    def context(self, tokens, index, history):
        tag_context = tuple(history[max(0, index - self._n + 1):index])
        return (tag_context, tokens[index])



