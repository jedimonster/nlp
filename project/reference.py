from nltk.corpus import reuters
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from features import TWSCalculator
from terminals import get_document_objects, WordTermExtractor

__author__ = 'itay'
if __name__ == '__main__':
    cats_limiter = categories = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'money-supply',
                                 'ship']  # top 8
    training_fileids = fileids = filter(lambda fileid: "training" in fileid and len(reuters.categories(fileid)) == 1,
                                        reuters.fileids(cats_limiter))

    training_documents = [" ".join(sum(reuters.sents(fid), [])) for fid in training_fileids]
    training_docs_categories = [reuters.categories(fid)[0] for fid in training_fileids]

    map(lambda x: x.lower, training_documents)

    training_documents_objects = get_document_objects(training_documents, training_docs_categories)
    # top 500
    # words = ['said', 'mln', 'vs', 'dlrs', 'lt', 'cts', '000', 'net', 'loss', 'pct', 'year', 'company', 'inc', 'shr',
    # 'billion', 'profit', 'share', 'corp', '1986', 'would', 'bank', 'shares', 'qtr', 'revs', 'stock', 'one',
    # 'oil', 'trade', 'group', 'two', ',"', 'also', '10', 'last', 'co', 'new', 'sales', 'march', 'april', '31',
    # 'note', 'quarter', 'offer', 'per', 'market', 'first', '15', 'oper', '1987', 'record', 'ltd', 'dividend',
    #          '1985', 'dlr', 'may', 'three', 'earnings', '20', '12', 'tax', '4th', 'japan', 'agreement', 'common',
    #          'sale', 'rate', '50', '30', 'exchange', 'international', 'six', 'board', '25', '."', 'avg', 'shrs', 'unit',
    #          'interest', 'five', 'pay', 'stake', '11', 'prior', 'acquisition', 'foreign', 'told', 'today', 'stg',
    #          'cash', '16', 'shareholders', 'government', 'american', 'div', 'operations', 'merger', 'banks', 'week',
    #          '13', '17', 'investment', 'price', 'includes', 'financial', 'prices', 'business', '28', 'nine', 'february',
    #          'total', 'could', 'january', 'buy', 'expected', '18', 'gain', 'rates', 'securities', '14', 'end', '>,',
    #          'world', 'four', 'rose', 'agreed', 'current', 'increase', 'japanese', 'made', 'companies', 'meeting',
    #          'split', 'earlier', 'general', 'qtly', 'sell', 'analysts', 'says', 'spokesman', 'assets', 'money',
    #          'dollar', 'months', 'president', 'results', 'capital', 'national', 'industries', 'added', 'ended', 'debt',
    #          'federal', 'management', 'official', '19', 'major', 'expects', 'statement', 'outstanding', 'industry',
    #          'chairman', 'canada', 'mths', 'sets', 'december', 'subsidiary', 'time', 'reported', 'products', 'bid',
    #          'growth', 'based', '22', 'month', 'opec', 'acquire', 'united', 'take', 'plans', '24', 'rise', 'announced',
    #          'operating', '27', 'credit', 'income', 'crude', 'cut', 'tender', 'jan', '40', 'purchase', 'trading',
    #          'terms', 'due', 'plc', '23', 'countries', 'firm', 'production', 'owned', 'years', 'profits', 'talks',
    #          'canadian', 'states', '35', 'commission', 'lower', 'term', 'eight', 'compared', 'higher', 'sees', 'full',
    #          '100', '26', 'economic', 'annual', 'period', 'plan', 'loan', 'west', 'fell', 'marks', 'officials',
    #          'markets', 'half', 'fiscal', 'previously', 'bpd', 'systems', 'since', 'report', 'feb', 'seven', 'gas',
    #          'exports', 'minister', '21', 'system', 'yesterday', 'day', 'energy', 'acquired', '1st', 'quarterly',
    #          'revenues', 'around', 'make', 'proposed', 'payable', 'part', 'currency', 'next', 'usair', 'sold', 'held',
    #          'long', 'takeover', 'excludes', '>.', 'second', '29', 'state', 'approval', 'well', 'extraordinary',
    #          'issue', 'francs', 'increased', 'savings', 'public', 'set', 'losses', 'include', 'control', 'fed', '500',
    #          'petroleum', 'sources', '60', 'policy', 'services', 'development', 'including', 'loans', '.,', 'high',
    #          'completed', '45', 'pacific', 'non', 'however', 'ago', 'value', 'baker', 'subject', 'deal', 'treasury',
    #          '34', '75', 'costs', 'trust', 'central', 'early', 'transaction', 'continue', 'payout', 'supply', 'banking',
    #          'prime', 'union', 'discontinued', 'deficit', 'imports', 'result', 'holdings', 'days', '33', '80', 'june',
    #          'fourth', 'third', 'finance', 'house', '51', 'york', 'south', 'surplus', 'gencorp', '--', 'industrial',
    #          'action', 'another', 'level', 'still', 'domestic', 'reuters', 'preferred', 'resources', 'approved',
    #          'already', 'bought', 'british', 'future', 'department', '37', 'reserves', 'bill', 'help', 'firms',
    #          'committee', 'charge', 'recent', 'insurance', 'average', 'figures', '32', 'fall', 'barrels', 'yen',
    #          'turnover', 'service', 'making', 'holders', 'saudi', 'move', 'holding', 'division', 'raised', 'proposal',
    #          'funds', 'cyclops', '42', 'twa', 'point', '38', 'might', 'reagan', 'brazil', '36', 'comment', 'chief',
    #          'stores', 'gains', 'declined', '3rd', '300', 'possible', 'change', '200', 'letter', 'open', 'certain',
    #          'close', 'demand', 'reserve', '55', 'previous', 'partners', 'largest', 'analyst', 'say', 'pretax',
    #          'strong', 'effective', 'much', 'cost', 'estimated', 'likely', 'executive', 'western', 'european',
    #          'commercial', '65', 'data', 'reduce', 'paid', 'pact', 'signed', 'directors', 'computer', 'equity', 'line',
    #          '43', '52', 'economy', 'distribution', 'fund', 'private', 'news', 'name', 'asked', 'final', 'declared',
    #          'association', 'pre', 'additional', 'ministry', 'given', 'several', 'country', '2nd', 'investors',
    #          'rights', 'texas', 'paris', 'restructuring', 'amount', 'ag', 'sector', 'shareholder', 'nations',
    #          'decision', 'issued', 'barrel', 'within', 'short', 'china', 'investor', 'currently', 'nil', 'output',
    #          'give', 'basis', 'gatt', 'financing', '44', 'raise', 'offered', '41', 'raises', 'ec', '47', 'gulf', 'ct',
    #          'german', 'north', '90', 'received', '64', 'security', '46', '70', 'budget', 'levels', 'dec', 'court',
    #          'called', 'respectively', 'lending', 'ecuador', 'late', '1988', '49', 'noted', 'real', 'dealers', '39']
    # words = ['said', 'mln', 'vs', 'dlrs', 'lt', 'cts', '000', 'net', 'loss', 'pct', 'year', 'company', 'inc', 'shr',
    #          'billion', 'profit', 'share', 'corp', '1986', 'would', 'bank', 'shares', 'qtr', 'revs', 'stock', 'one',
    #          'oil', 'trade', 'group', 'two', ',"', 'also', '10', 'last', 'co', 'new', 'sales', 'march', 'april', '31',
    #          'note', 'quarter', 'offer', 'per', 'market', 'first', '15', 'oper', '1987', 'record', 'ltd', 'dividend',
    #          '1985', 'dlr', 'may', 'three', 'earnings', '20', '12', 'tax', '4th', 'japan', 'agreement', 'common',
    #          'sale', 'rate', '50', '30', 'exchange', 'international', 'six', 'board', '25', '."', 'avg', 'shrs', 'unit',
    #          'interest', 'five', 'pay', 'stake', '11', 'prior', 'acquisition', 'foreign', 'told', 'today', 'stg',
    #          'cash', '16', 'shareholders', 'government', 'american', 'div', 'operations', 'merger', 'banks', 'week',
    #          '13', '17', 'investment', 'price', 'includes', 'financial', 'prices', 'business', '28', 'nine', 'february',
    #          'total', 'could', 'january', 'buy', 'expected', '18', 'gain', 'rates', 'securities', '14', 'end', '>,',
    #          'world', 'four', 'rose', 'agreed', 'current', 'increase', 'japanese', 'made', 'companies', 'meeting',
    #          'split', 'earlier', 'general', 'qtly', 'sell', 'analysts', 'says', 'spokesman', 'assets', 'money',
    #          'dollar', 'months', 'president', 'results', 'capital', 'national', 'industries', 'added', 'ended', 'debt',
    #          'federal', 'management', 'official', '19', 'major', 'expects', 'statement', 'outstanding', 'industry',
    #          'chairman', 'canada', 'mths', 'sets', 'december', 'subsidiary', 'time', 'reported', 'products', 'bid',
    #          'growth', 'based', '22', 'month', 'opec', 'acquire', 'united', 'take', 'plans', '24', 'rise', 'announced',
    #          'operating', '27', 'credit', 'income', 'crude', 'cut', 'tender', 'jan', '40', 'purchase', 'trading',
    #          'terms', 'due', 'plc', '23', 'countries', 'firm', 'production', 'owned', 'years', 'profits', 'talks',
    #          'canadian', 'states', '35', 'commission', 'lower', 'term', 'eight', 'compared', 'higher', 'sees', 'full',
    #          '100', '26', 'economic', 'annual', 'period', 'plan', 'loan', 'west', 'fell', 'marks', 'officials',
    #          'markets', 'half', 'fiscal', 'previously', 'bpd', 'systems', 'since', 'report', 'feb', 'seven', 'gas',
    #          'exports', 'minister', '21', 'system', 'yesterday', 'day', 'energy', 'acquired', '1st', 'quarterly',
    #          'revenues', 'around', 'make', 'proposed', 'payable', 'part', 'currency', 'next', 'usair', 'sold', 'held',
    #          'long', 'takeover', 'excludes', '>.', 'second', '29', 'state', 'approval', 'well', 'extraordinary',
    #          'issue', 'francs', 'increased', 'savings', 'public', 'set', 'losses', 'include', 'control', 'fed', '500',
    #          'petroleum', 'sources', '60', 'policy', 'services', 'development', 'including', 'loans', '.,', 'high',
    #          'completed', '45', 'pacific', 'non', 'however', 'ago', 'value', 'baker', 'subject', 'deal', 'treasury',
    #          '34', '75', 'costs', 'trust', 'central', 'early', 'transaction', 'continue', 'payout', 'supply', 'banking',
    #          'prime', 'union', 'discontinued', 'deficit', 'imports', 'result', 'holdings', 'days', '33', '80', 'june',
    #          'fourth', 'third', 'finance', 'house', '51', 'york', 'south', 'surplus', 'gencorp', '--', 'industrial',
    #          'action', 'another', 'level', 'still', 'domestic', 'reuters', 'preferred', 'resources', 'approved',
    #          'already', 'bought', 'british', 'future', 'department', '37', 'reserves', 'bill', 'help', 'firms',
    #          'committee', 'charge', 'recent', 'insurance', 'average', 'figures', '32', 'fall', 'barrels', 'yen',
    #          'turnover', 'service', 'making', 'holders', 'saudi', 'move', 'holding', 'division', 'raised', 'proposal',
    #          'funds', 'cyclops', '42', 'twa', 'point', '38', 'might', 'reagan', 'brazil', '36', 'comment', 'chief',
    #          'stores', 'gains', 'declined', '3rd', '300', 'possible', 'change', '200', 'letter', 'open', 'certain',
    #          'close', 'demand', 'reserve', '55', 'previous', 'partners', 'largest', 'analyst', 'say', 'pretax',
    #          'strong', 'effective', 'much', 'cost', 'estimated', 'likely', 'executive', 'western', 'european',
    #          'commercial', '65', 'data', 'reduce', 'paid', 'pact', 'signed', 'directors', 'computer', 'equity', 'line',
    #          '43', '52', 'economy', 'distribution', 'fund', 'private', 'news', 'name', 'asked', 'final', 'declared',
    #          'association', 'pre', 'additional', 'ministry', 'given', 'several', 'country', '2nd', 'investors',
    #          'rights', 'texas', 'paris', 'restructuring', 'amount', 'ag', 'sector', 'shareholder', 'nations',
    #          'decision', 'issued', 'barrel', 'within', 'short', 'china', 'investor', 'currently', 'nil', 'output',
    #          'give', 'basis', 'gatt', 'financing', '44', 'raise', 'offered', '41', 'raises', 'ec', '47', 'gulf', 'ct',
    #          'german', 'north', '90', 'received', '64', 'security', '46', '70', 'budget', 'levels', 'dec', 'court',
    #          'called', 'respectively', 'lending', 'ecuador', 'late', '1988', '49', 'noted', 'real', 'dealers', '39',
    #          'number', 'adjusted', 'special', 'purolator', 'taft', 'goods', 'way', 'washington', 'french', 'export',
    #          'filing', 'low', 'following', 'forecast', '48', 'air', 'credits', 'effect', 'class', 'equipment',
    #          'disclosed', '400', 'whether', 'less', 'conditions', 'monetary', 'seek', 'members', 'conference', 'dixons',
    #          '85', 'undisclosed', 'least', 'america', '600', 'balance', '87', 'free', 'september', 'technology',
    #          'secretary', 'good', 'seeking', '53', 'base', 'reached', 'provide', 'mark', 'payment', 'must', 'friday',
    #          'worth', '150', 'used', 'deposits', '72', 'decline', '800', 'among', '88', 'units', 'right', 'back',
    #          'joint', 'buys', 'life', 'large', 'england', 'administration', 'bond', 'taiwan', 'need', 'think', 'far',
    #          'negotiations', 'southern', 'payments', 'periods', 'related', 'work', 'initial', 'cuts', 'discount',
    #          'later', 'many', 'power', 'see', 'diluted', 'receive', 'selling', 'parent', 'tariffs', 'measures',
    #          'planned', 'de', 'chemical', 'according', 'commerce', 'properties', 'offering', 'remain', 'currencies',
    #          'exploration', 'included', 'revised', 'better', '54', 'piedmont', 'date', 'program', 'order', 'accord',
    #          'provision', 'position', 'chrysler', 'hold', 'communications', 'interests', 'partnership', 'regular',
    #          'ending', '57', 'senior', 'july', 'acquisitions', 'although', '56', '68', 'support', 'sterling', 'details',
    #          'earned', 'owns', 'venture', 'semiconductor', 'cable', 'reporters', 'community', 'target', 'germany',
    #          'show', 'bankers', '58', 'led', 'closing', 'changes', 'issues', '63', 'saying', 'weeks', '00', 'port',
    #          'agency', 'plant', 'available', 'buyout', 'agreements', 'meet', 'businesses', 'reduced', 'return',
    #          'express', 'arabia', 'estate', 'kong', 'response', 'come', 'range', 'drop', '700', 'get', 'cents',
    #          'shearson', 'sells', '61', 'reflect', 'bills', 'pressure', 'dividends', 'london', '86', 'product', '69',
    #          'charges', 'even', 'expenses', '66', '83', 'economists', 'completes', '62', 'without', '71', 'city',
    #          'france', 'despite', '59', 'officer', 'scheduled', 'closed', 'hong', 'ministers', 'holds', 'standard',
    #          'option', 'addition', 'james', 'november', 'latest', 'go', 'home', 'continuing', 'fixed', 'director',
    #          'use', 'shipping', 'information', 'caesars', 'import', 'main', 'stocks', 'vice', 'notes', 'dispute',
    #          'combined', 'consider', 'operation', 'investments', 'plus', 'makes', 'become', 'quoted', 'food', 'revenue',
    #          'reduction', 'problems', '95', 'congress', 'taken', 'california', 'hughes', '77', '92', 'hutton', 'past',
    #          'account', 'warner', 'present', 'member', 'buying', '),', 'continued', 'allow', 'steel', 'repurchase',
    #          'tomorrow', 'area', 'traders', '81', 'research', 'impact', 'voting', 'put', 'principle', 'review',
    #          'bundesbank', 'crowns', 'intent', 'consolidated', 'telecommunications', 'brown', 'decided', 'efforts',
    #          '67', '94', 'going', 'small', 'bp', 'increases', 'soon', '73', 'yeutter', 'medical', 'form', 'telephone',
    #          '74', 'pipeline', 'corporate', 'people', 'improved', 'convertible', 'little', 'chemlawn', 'australian',
    #          'settlement', 'expect', '78', '89', '82', 'manufacturing', 'liquidity', 'adding', 'round', 'limited',
    #          'regulatory', 'television', 'remaining', 'key', 'recently', 'strike', 'us', 'tokyo', 'airlines', 'orders',
    #          'allied', '76', 'workers', 'provisions', '99', 'act', 'october', 'competition', 'borg', 'study',
    #          'definitive', 'proposals', '900', 'controls', 'capacity', 'gave', 'place', 'announcement', 'hit', 'europe',
    #          'press', 'reports', 'accounting', 'boost', 'royal', 'majority', 'mortgage', 'britain', '93', 'wants',
    #          'law', 'unchanged', 'near', 'offers', 'refinery', 'electronics', 'allegheny', 'concern', 'fully', '...',
    #          'daily', 'contract', 'filed', 'needed', 'australia', 'post', 'leading', 'seen', 'application', 'sosnoff',
    #          '79', 'significant', 'continental', 'increasing', 're', 'paper', 'start', 'property', 'required',
    #          'attempt', 'taking', 'proceeds', 'options', 'health', 'probably', 'immediately', 'korea', '98', 'former',
    #          'existing', 'interview', 'exclude', 'growing', 'lost', 'performance', 'sharp', 'working', 'complete',
    #          'restated', 'gold', 'marketing', 'call', 'yet', 'dutch', 'field', 'purchased', 'wholly', 'mid', 'like',
    #          'estimates', 'row', 'electric', 'showed', 'east', '120', 'began', '91', 'video', 'customer', 'waste',
    #          'office', 'completion', 'expansion', 'force', 'extended', 'believe', 'rev', 'areas', 'sea', 'accounts',
    #          'morning', 'considering', 'greater', 'shell', 'local', 'best', '250', 'dome', 'venezuela', 'sec', 'ends',
    #          'courier', 'benefit', 'overseas', 'block', 'necessary', 'stability', 'activities', 'bancorp', 'formed',
    #          'boston', 'merge', 'rejected', 'discuss', 'street', 'august', 'changed', 'valued', 'st', 'sharply',
    #          'legislation', 'war', 'mining', 'writedown', 'flow', 'vote', 'ab', 'interested', 'amc', 'quota',
    #          'transportation', 'cyacq', 'mines', 'want', 'rises', 'means', 'problem', '125', 'harper', 'volume',
    #          'nakasone', 'santa', 'great', 'maker', 'ordinary', 'assistance', 'substantial', 'consumer', 'purchases',
    #          'longer', 'mainly', 'provided', 'political', 'customers', 'ships', 'pending', 'warrants', '96', 'natural',
    #          'san', 'discussions', 'futures', 'heavy', 'yr', 'similar', 'sanctions']

    #top IG r8:
    words = set(["today", "tankers", "bid", "shr", "sell", "agreed", "de", "cargo", "prior", "4th", "today", "westminster",
             "cts", "fed", "foreign", "tender", "prior", "base", "shares", "tariffs", "bank", "association", "banks",
             "minister", "retaliation", "inc", "latest", "borrowings", "customer", "cts", "spot", "loss", "major",
             "intelligence", "treasury", "cutting", "move", "strike", "oper", "vs", "quarterly", "buy", "kuwait",
             "target", "shr", "fall", "bilateral", "refineries", "wti", "brazil", "state", "government", "net",
             "ecuador", "company", "loss", "profit", "circulation", "sanctions", "production", "vs", "growth", "bills",
             "revs", "subsidiary", "export", "market", "petroleos", "rate", "refinery", "acquisition", "protest",
             "provisional", "ranges", "net", "terms", "exchequer", "accord", "banking", "commission", "31", "says",
             "div", "contract", "house", "federal", "div", "sea", "quota", "redundancies", "tonnes", "barrels",
             "drilling", "buy", "inc", "merger", "dealers", "liquidity", "000", "nations", "provided", "broad",
             "immediately", "crude", "term", "year", "money", "representative", "disclosed", "inc", "net", "bank",
             "stock", "vs", "4th", "reserve", "grew", "purchase", "rise", "yards", "market", "group", "completes",
             "japan", "assistance", "stake", "split", "qtr", "company", "jan", "shr", "qtr", "acquisition", "qtr",
             "window", "revs", "shrs", "dollars", "central", "agreement", "gulf", "economic", "ship", "janeiro",
             "deficit", "mln", "dlrs", "treasury", "venezuela", "m2", "lt", "share", "03", "trade", "bpd", "banks",
             "tokyo", "lt", "coast", "exchange", "consumption", "lt", "output", "official", "minister", "company",
             "net", "payout", "workers", "intervention", "pct", "yeutter", "last", "cuts", "prices", "investor", "qtr",
             "monetary", "31", "qtr", "unfair", "lt", "3rd", "agreement", "france", "paris", "maritime", "week", "vs",
             "net", "4th", "qtly", ">,", "grade", "note", "foreign", "congress", "effective", "shareholders", "baker",
             "japanese", "rates", "says", "completed", "federal", "intermediate", "agreed", "vs", "raises", "acquired",
             "net", "cts", "united", "citibank", "broadly", "interest", "help", "price", "follows", "imports", "inc",
             "record", "crew", "inc", "revs", "transaction", "currencies", "nations", "deposits", "undisclosed",
             "trade", "shr", "point", "31", "maturity", "inc", "barrel", "legislation", "rises", "1st", "loss",
             "percentage", "economic", "told", "finance", "000", "striking", "deposit", "cts", "day", "agreement",
             "market", "cts", "buys", "revs", "employers", "shr", "includes", "lending", "shipping", "stabilise",
             "monetary", "exploration", "markets", "transactions", "avg", "labour", "cut", "energy", "bid", "week",
             "reagan", "transport", "chinese", "member", "said", "acquire", "almir", "shrs", "vs", "ports", "stocks",
             "pay", "qtly", "fnv", "1986", "shipbuilding", "offer", "strikes", "impose", "union", "compared", "short",
             "oil", "country", "fundamentals", "unadjusted", "dumping", "stop", "unit", "qtr", "supply", "exports",
             "shr", "rate", "corp", "said", "mths", "net", "vessel", "adjusted", "average", "open", "shares", "revised",
             "vs", "canal", "seasonally", "protectionist", "dollar", "profit", "harbour", "states", "acquire",
             "dispute", "approval", "seamen", "bundesbank", "system", "sell", "compares", "note", "gas", "official",
             "england", "said", "avg", "lt", "dealers", "chase", "january", "nwbl", "narrowly", "1986", "earthquake",
             "fed", "rotterdam", "goods", "half", "would", "maturing", "discount", "around", "missiles", "saudi",
             "firm", "bankers", "bbl", "fell", "pct", "cash", "dwt", "semiconductors", "profit", "opec", "defined",
             "takeover", "prior", "cts", "corp", "pct", "practices", "cts", "exchange", "february", "qtr", "pipeline",
             "postings", "securities", "000", "share", "arabia", "bank", "vs", "currency", "texas", "common", "ceiling",
             "lt", "vessels", "offer", "ferry", "rose", "shipowners", "shortage", "producing", "spokesman", "revs",
             "chequable", "band", "semiconductor", "work", "dividend", "today", "money", "note", "payable", "port",
             "share", "petroleum", "rio", "record", "terms", "said", "miyazawa", "money", "loss", "herald", "price",
             "south", "company", "acceptances", "management", "talks", "revs", "inc", "loans", "retaliate", "stg",
             "followed", "qtr", "shr", "money", "m1", "subject", "billion", "revs", "funds", "cts", "enterprise",
             "previous", "protectionism", "uruguay", "falls", "tomorrow", "washington", "year", "sea", "net",
             "industry", "forecast", "corp", "cut", "intervene", "note", "profit", "m3", "pumping", "000", "ships",
             "000", "last", "lt", "mths", "shr", "merger", "gatt", "al", "excludes", "day", "officials", "earnings",
             "stake", "germany", "world", "brazilian", "surplus", "corp", "dividend", "stability", "grades",
             "outstanding", "prime", "bank", "december", "dlrs", "minister", "results", "ministry", "benchmark",
             "import", "group", "revised", "shares", "rates", "corp", "countries", "would", "government", "points"]
    )
    # tws_calculator = TWSCalculator(training_documents_objects, training_docs_categories)
    # word_term_extractor = WordTermExtractor(training_documents_objects, tws_calculator)
    #
    # top_terms = word_term_extractor.top_common_words(500)
    print training_documents[0]
    print training_fileids

    vectorizer = TfidfVectorizer(input='content', max_features=500, vocabulary=words, stop_words=None)
    feature_matrix = vectorizer.fit_transform(training_documents)

    classifier = OneVsRestClassifier(MultinomialNB())
    classifier.fit(feature_matrix, training_docs_categories)


    # Test:
    test_fileids = fileids = filter(lambda fileid: "training" not in fileid and len(reuters.categories(fileid)) == 1,
                                    reuters.fileids(cats_limiter))
    test_documents = [" ".join(sum(reuters.sents(fid), [])) for fid in test_fileids]
    test_docs_categories = [reuters.categories(fid)[0] for fid in test_fileids]
    map(lambda x: x.lower, test_documents)

    test_features = vectorizer.transform(test_documents)

    predictions = classifier.predict(test_features)

    metrics = sklearn.metrics.precision_recall_fscore_support(test_docs_categories, predictions, average='weighted')

    print "Metrics (percision, recall, fmeasure):", metrics

    accuracy = accuracy_score(test_docs_categories, predictions)

    print "Accuracy:", accuracy
