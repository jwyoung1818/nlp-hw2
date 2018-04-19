#!/usr/bin/
import numpy as np
import sys
def extractFeatures(filename):
  f = open(filename, 'r')
  features = []
  fea_hash = {}
  for line in f:
    word_label = line.strip().split('\t')
    label = int(word_label[1])
    words = word_label[0].split(' ')
    for word in words:
      feature = (word, label)
      #print feature
      if feature not in features:
        features.append(feature)
        fea_hash[feature] = len(features) - 1
  print len(features)
  f.close
  return features, fea_hash
def sen2vec(filename, features, fea_hash, l = None):
  f = open(filename, 'r')
  sen_vectors = []
  for line in f:
    word_label = line.strip().split('\t')
    label = int(word_label[1])
    words = word_label[0].split(' ')
    sen_vectors.append((words,label))
  f.close()
  return sen_vectors
def s2vec(sentence, features, fea_hash, l):
  s_vec = [0.0 for _ in xrange(0, len(features))]
  for word in sentence:
    feature = (word, l)
    if feature in fea_hash:
      s_vec[fea_hash[feature]] += 1
  s_vec = np.array(s_vec)
  return s_vec
def ss2vec(ss, features, fea_hash):
  ss_vec = [[0.0 for _ in xrange(0, 3)] for _ in xrange(0, len(ss))]
  for index in xrange(0, len(ss)):
    sentence = ss[index][0]
    for c in xrange(0, 3):
      ss_vec[index][c] = s2vec(sentence, features, fea_hash, c)
  return ss_vec
def train(filename, features, fea_hash, epoch, learning_rate, dev_filename, h=False, step=10000):
  # extract features firstly
  outfile = 'accuracy'
  if h:
    outfile += '.hinge'
  output = open(outfile, 'w+')
  train_sentences = sen2vec(filename, features, fea_hash)
  dev_sentences = sen2vec(dev_filename, features, fea_hash)
  weight = [[0.0 for _ in xrange(0, len(features))] for _ in xrange(0,3)]
  weight = np.array(weight)
  weights = []
  train_ss = ss2vec(train_sentences, features, fea_hash)
  #dev_ss = ss2vec(dev_sentences, features, fea_hash)
  for i in xrange(0, epoch): 
    #print i, len(train_sentences)
    for index in xrange(0, len(train_sentences)):
      sentence_label = train_sentences[index]
      sentence = sentence_label[0]
      sentence = np.array(sentence)
      label = sentence_label[1]
      argmax, predicated = 0, 0
      for c in xrange(0,3):
        #s_vec = s2vec(sentence, features, fea_hash, c)
        s_vec = train_ss[index][c]
        cv = np.dot(s_vec, weight[c])
        if h:
          if c != label:
            cv += 1
        #print "cv is:", cv, 
        if cv >= argmax:
          argmax = cv
          predicated = c
      if label != predicated:
        #print "update", i, argmax, label, predicated, train_sentences.index(sentence_label)
        s_vec = s2vec(sentence, features, fea_hash, label)
        weight[label] += s_vec * learning_rate
        weight[predicated] -= s_vec * learning_rate
      if index / step > 0 and index % step == 0:
        out = 'output/' + str(i) + '-' + str(index/step) + '.out'
        if h:
          out = out + '.hinge'
        np.save(out, weight)
        re, wrongs = predict_sens(weight, dev_sentences, features, fea_hash)
        print "accuracy", index, re
        output.write(str(i) + "-" + str(index/step) + " "  + str(re) + "\n")
    out = 'output/' + str(i) + '.out'
    if h:
      out = out + '.hinge'
    np.save(out, weight)
    re, wrongs = predict_sens(weight, dev_sentences, features, fea_hash)
    print "accuracy", re
    output.write(str(i) + " "  + str(re) + "\n")
  return weight

def predict(weight, s, ss_vec_i):
  argmax, predicated = 0, 0
  #sentence = np.array(sentence)
  for c in xrange(0, 3):
    sentence = ss_vec_i[c] 
    cv = np.dot(sentence, weight[c])
    if cv >= argmax:
      argmax, predicated = cv, c
  return predicated
def predict_sens(weight, sentences, features, fea_hash):
  total = len(sentences)
  right = 0
  wrong = []
  ss_vec = ss2vec(sentences, features, fea_hash)
  for i  in xrange(0, len(sentences)):
    sentence_label = sentences[i]
    s = sentence_label[0]
    label = sentence_label[1]
    predicated = predict(weight, s, ss_vec[i])
    #print predicated
    if predicated == label:
      right += 1
    else:
       wrong.append((sentences[i], predicated))
  return float(right)/total * 100.0, wrong
def evaluate(weight_file, features, fea_hash, dev_filename):
   print "largest weight features"
   weight = np.load(weight_file)
   features = np.array(features)
   ws = [[],[],[]]
   for i in xrange(0, len(weight)):
     w = weight[i]
     for j in xrange(0, len(w)):
       print features[j][1]
       index = int(features[j][1])
       ws[index].append((features[j][0], w[j]))
     print i
     arg = np.argsort(w)
     indices = arg[len(arg)-10: len(arg)]
     indices = arg[0:10]
     #print indices
     top_f = features[indices]
     #print "top-10"
     #print top_f, w[indices]
   ws = np.array(ws)
   for i in xrange(0, len(ws)):
     w = ws[i]
     print i
     to_sort = w[:, 1].astype(np.float)
     arg = np.sort(to_sort)
     indices = arg[len(arg)-10: len(arg)]
     top_f = features[indices]
     print "top-10"
     print top_f, w[indices]
   dev_sentences = sen2vec(dev_filename, features, fea_hash) 
   re, wrongs = predict_sens(weight, dev_sentences, features, fea_hash)
   print re
   for wrong in wrongs:
     print wrong
   return re, wrongs
def main():
  train_filename = 'sst3/sst3.train'
  dev_filename = 'sst3/sst3.dev'
  dev_test_filename = 'sst3/sst3.devtest'
  full_filename = 'sst3/sst3.train-full-sentences'
 # train_filename = full_filename 
  #train_filename = dev_test_filename
  epoch = 20
  #epoch = 1
  learning_rate = 0.01
  #learning_rate = 1
  features, fea_hash = extractFeatures(train_filename)
  if len(sys.argv) > 1 and sys.argv[1] == 'h':
    weight = train(train_filename, features, fea_hash, epoch, learning_rate, dev_filename, True) 
  elif len(sys.argv) > 2 and sys.argv[1] == 'e':
    weight_file = sys.argv[2]
    evaluate(weight_file, features, fea_hash, dev_test_filename)
  else:
    weight = train(train_filename, features, fea_hash, epoch, learning_rate, dev_filename)
  #dev_sentences = sen2vec(dev_filename, features, fea_hash)
  
if __name__ == '__main__':
  main()
