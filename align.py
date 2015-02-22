#!/usr/bin/env python
import optparse
import sys
import string
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

# sys.stderr.write("Training with Dice's coefficient...")
# bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
# f_count = defaultdict(int)
# e_count = defaultdict(int)
# fe_count = defaultdict(int)
# for (n, (f, e)) in enumerate(bitext):
#   for f_i in set(f):
#     f_count[f_i] += 1
#     for e_j in set(e):
#       fe_count[(f_i,e_j)] += 1
#   for e_j in set(e):
#     e_count[e_j] += 1
#   if n % 500 == 0:
#     sys.stderr.write(".")

# dice = defaultdict(int)
# for (k, (f_i, e_j)) in enumerate(fe_count.keys()):
#   dice[(f_i,e_j)] = 2.0 * fe_count[(f_i, e_j)] / (f_count[f_i] + e_count[e_j])
#   if k % 5000 == 0:
#     sys.stderr.write(".")
# sys.stderr.write("\n")

# for (f, e) in bitext:
#   for (i, f_i) in enumerate(f): 
#     for (j, e_j) in enumerate(e):
#       if dice[(f_i,e_j)] >= opts.threshold:
#         sys.stdout.write("%i-%i " % (i,j))
#   sys.stdout.write("\n")


sys.stderr.write("Training with IBM Model 1...")
bitext = []
sent_count = 0
# bitext = [[sentence.lower().strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
for (f,e) in zip(open(f_data), open(e_data)):
  if sent_count < opts.num_sents:
    bitext.append([[None] + f.lower().strip().split(), e.lower().strip().split()])
  else:
    break
  sent_count += 1

# for (f, e) in zip(open(f_data), open(e_data)):
#   bitext.append([[None] + f.lower().strip().split(), e.lower().strip().split()])

total_f = defaultdict(int)
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)
t_ef = defaultdict(float)
s_total = defaultdict(int)

for (n, (f, e)) in enumerate(bitext):
  for e_i in set(e):
    e_count[e_i] += 1
    for f_j in set(f):
      fe_count[(e_i,f_j)] = 0
      f_count[f_j] += 1
    
total_foreign_words = len(f_count)
for (n, (f, e)) in enumerate(bitext):
  for (j, english_word) in enumerate(e):
    for (i, foreign_word) in enumerate(f):
      #sys.stderr.write(str((m,n)) +" ")
      t_ef[(english_word, foreign_word)] = 1.0 / (total_foreign_words + 1)  #initialize t(e|f) uniformly 

convergence = 15
converged = False
prev_sum = sum(t_ef.values())

# while not converged:
while convergence > 0:
  # initialize
  for (n, (f, e)) in enumerate(bitext):
    for e_i in set(e):
      for f_j in set(f):
        total_f[f_j] = 0    #total(f) = 0 for all words f
        fe_count[(e_i,f_j)] = 0   #count(e|f) = 0 for all words e,f

  for (n, (f, e)) in enumerate(bitext):   # for all sentence pairs (eng_sentence, foreign_sentence) do:
    #compute normalization
    for e_i in e:  #for all words e in eng_sentence do
      s_total[e_i] = 0  #s-total[e] = 0
      for f_j in f:#for all words f in foreign_sentence do:
        s_total[e_i] += t_ef[(e_i, f_j)]  #s-total(e) += t(e|f)

    # collect counts
    for e_i in e: #for all words e in eng_sentence do
      for f_j in f: #for all words f in foreign_sentence do
        fe_count[(e_i, f_j)] += t_ef[(e_i, f_j)] / float(s_total[e_i]) #count(e|f) += t(e|f) / s-total(e)
        total_f[f_j] += t_ef[(e_i, f_j)] / float(s_total[e_i]) #total(f) += t(e|f) / s-total(e)
    if n % 500 == 0:
      sys.stderr.write(".")

  #estimate probabilities
  # for f_j in total_f: #for all foreign words f do:
  #   for e_i in s_total: #for all english words e do:
  #     t_ef[(e_i, f_j)] = fe_count[(e_i, f_j)] / float(total_f[f_j])  #t(e|f) = count(e|f) / total(f)
  for (n, (f, e)) in enumerate(bitext):
    for f_j in set(f): #for all foreign words f do:
      for e_i in set(e): #for all english words e do:
        t_ef[(e_i, f_j)] = fe_count[(e_i, f_j)] / float(total_f[f_j])  #t(e|f) = count(e|f) / total(f)

  convergence -= 1
  # if (abs(prev_sum - sum(t_ef.values())) < 0.000001):
  #   # print abs(prev_sum - sum(fe_transition.values()))
  #   converged = True
  # else:
  #   prev_sum = sum(t_ef.values())

#################################################################################################################################
#IBM Model 2 Implementation
# bitext = [[sentence.lower().strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

# for (n, (f, e)) in enumerate(bitext):
#   for e_i in set(e):
#     e_count[e_i] += 1
#   for f_j in set(f):
#     f_count[f_j] += 1

# total_foreign_words = len(f_count)
# for (m, english_word) in enumerate(e_count):
#   for (n, foreign_word) in enumerate(f_count):
#     t_ef[(english_word, foreign_word)] = 1.0 / (total_foreign_words + 1)  #initialize t(e|f) uniformly 

# sent_count = 0
# for (f,e) in zip(open(f_data), open(e_data)):
#   if sent_count < opts.num_sents:
#     bitext.append([[None] + f.lower().strip().split(), e.lower().strip().split()])
#   else:
#     break
#   sent_count += 1

prev_sum = sum(t_ef.values())

align = defaultdict(float)
for (f, e) in bitext:
  for (i, f_i) in enumerate(f): 
    for (j, e_j) in enumerate(e):
      align[(i, j, len(e), len(f))] = 1.0 / (len(f) + 1)

count_a = defaultdict(int)
total_a = defaultdict(int)

convergence = 15
converged = False

# while convergence > 0:
while not converged:
  for (n, (f, e)) in enumerate(bitext):
    for (i, f_i) in enumerate(f): 
      total_f[f_i] = 0
      for (j, e_j) in enumerate(e):
        fe_count[(e_j,f_i)] = 0
        count_a[(i, j, len(e), len(f))] = 0
        total_a[(j, len(e), len(f))] = 0

  for (n, (f, e)) in enumerate(bitext):
    le = len(e)
    lf = len(f)

    #computer normalization
    for (j, e_j) in enumerate(e):
      s_total[e_j] = 0
      for (i, f_i) in enumerate(f):
        s_total[e_j] += t_ef[(e_j, f_i)] * align[(i, j, le, lf)]

    for (j, e_j) in enumerate(e):
      for (i, f_i) in enumerate(f):
        c = (t_ef[(e_j, f_i)] * align[(i, j, le, lf)]) / float(s_total[e_j])
        fe_count[(e_j, f_i)] += c
        total_f[f_i] += c
        count_a[(i, j, le, lf)] += c
        total_a[(j, le, lf)] += c

  #estimate probabilities
  for (f, e) in bitext:
    for (i, f_i) in enumerate(f): 
      for (j, e_j) in enumerate(e):
        t_ef[(e_j, f_i)] = 0
        align[(i, j, len(e), len(f))] = 0

  for (f, e) in bitext:
    for (i, f_i) in enumerate(f): 
      for (j, e_j) in enumerate(e):
        t_ef[(e_j, f_i)] = fe_count[(e_j, f_i)] / float(total_f[f_i])
        align[(i, j, len(e), len(f))] = count_a[(i, j, len(e), len(f))] / float(total_a[(j, len(e), len(f))])

  if (abs(prev_sum - sum(t_ef.values())) < 0.00001):
    # print abs(prev_sum - sum(fe_transition.values()))
    converged = True
  else:
    prev_sum = sum(t_ef.values())
  convergence -= 1
#################################################################################################################################

for (f, e) in bitext:
  for (i, f_i) in enumerate(f): 
    for (j, e_j) in enumerate(e):
      #if t_ef[(e_j, f_i)] * align[(i, j, len(e), len(f))] >= opts.threshold and dice[(f_i,e_j)] >= opts.threshold:
      #if t_ef[(e_j, f_i)] * align[(i, j, len(e), len(f))] >= opts.threshold and i != 0:
      #if t_ef[(e_j, f_i)] >= opts.threshold and dice[(f_i,e_j)] >= opts.threshold:
      if t_ef[(e_j, f_i)] >= opts.threshold and i != 0:
        #sys.stderr.write(str(t_ef[(e_j, f_i)]))
        sys.stdout.write("%i-%i " % (i - 1,j))
  sys.stdout.write("\n")

# for (f, e) in bitext:
#   for (i, f_i) in enumerate(f):
#     best_p = 0
#     best_a = -1
#     for (j, e_j) in enumerate(e):
#       p = t_ef[(e_j, f_i)]
#       if p > best_p:
#         best_p = p
#         best_a = j
#     if best_a > -1:
#       sys.stdout.write("%i-%i " % (i, best_a))
#   sys.stdout.write("\n")

# for (f, e) in bitext:
#   for (j, e_j) in enumerate(e):
#     best_p = 0
#     best_a = -1
#     for (i, f_i) in enumerate(f):
#       p = t_ef[(e_j, f_i)]
#       if p > best_p:
#         best_p = p
#         best_a = i
#     if best_a > -1:
#       sys.stdout.write("%i-%i " % (best_a, j))
#   sys.stdout.write("\n")