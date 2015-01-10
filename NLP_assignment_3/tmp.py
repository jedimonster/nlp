__author__ = 'itay'
treebank = LazyCorpusLoader('treebank/combined', BracketParseCorpusReader, r'wsj_.*\.mrg')
trees = treebank.parsed_sents()
trees = map(filter_tree, trees)
prods = sum([list(tree_to_productions_parents(t, 'DUKIS')) for t in trees], list())

# only LHS = NP, no parent restriction:
nps = filter(lambda prod: prod[1].lhs().symbol() == 'NP', list(prods))
rhs = map(lambda x: x[1], nps)
rhs_dist = nltk.FreqDist(rhs)
print len(rhs_dist.keys())
print rhs_dist.tabulate()

# only LHS = NP below S
nps_below_s = map(lambda x: x[1],
                  filter(lambda prod: prod[0] == 'S' and prod[1].lhs().symbol() == 'NP', list(prods)))
nps_below_s_freq = nltk.FreqDist(nps_below_s)
# print len(set(nps_below_s))
# print nps_below_s[:10]

# only LHS = NP below VP
nps_below_vp = map(lambda x: x[1],
                   filter(lambda prod: prod[0] == 'VP' and prod[1].lhs().symbol() == 'NP', list(prods)))
nps_below_vp_freq = nltk.FreqDist(nps_below_vp)

nps_below_s_dist = nltk.MLEProbDist(nps_below_s_freq)
nps_below_vp_dist = nltk.MLEProbDist(nps_below_vp_freq)

samples = set(nps_below_s_dist.samples() + nps_below_vp_dist.samples())
rhs_probs = [(nps_below_s_dist.prob(rhs), nps_below_vp_dist.prob(rhs)) for rhs in samples]

print "NP BELOW S RHS:"
print nps_below_s_dist.samples()

print '--------'
print 'NP BELOW VP RHS:'
print  nps_below_vp_dist.samples()

print sum([x[0] != 0 and x[1] != 0 for x in rhs_probs])

print kl_divergence(rhs_probs)
