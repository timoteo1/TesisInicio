from __future__ import division
from collections import Counter, defaultdict
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
import functools

def readFile(fname):
    X = list()
    y = list()
    try:
    #with open(fname, 'r') as fdata:
        fdata = open(fname, 'rb')
        for l in fdata:
            l = l.strip()
            if l == '<review>':
                text = ''
                rating = None
                next_rating = False
                next_text = False
            elif l == '<rating>':
                next_rating = True
            elif l == '</rating>':
                next_rating = False
            elif l == '<review_text>':
                next_text = True
            elif l == '</review_text>':
                next_text = False
            elif l == '</review>':
                text = text.strip()
                if len(text) > 0 and rating is not None:
                    X.append(text)
                    d = [0]*5
                    assert rating > 0, "Rating 0"
                    d[rating-1] = 1
                    y.append(d)
            else:
                if next_rating:
                    rating = int(float(l))
                elif next_text:
                    text += l + ' '
        fdata.close()
    except Exception as e:
        print(e)
        quit()
    #print l
    return X, y

#divide una frase en una lista de palabras
def vocabulary(sentences):    
    words = set()
    frequency = Counter()
    for x in sentences:
        sentence_words = set()
        for w in x:
            words.add(w)
            if w not in sentence_words:
                sentence_words.add(w)
                frequency[w] += 1
    return words, frequency


def filter_vocabulary(words, frequency, total, min_value=0.0, max_value=1.0):
    assert min_value < max_value, "Values error"   #si no se cumple 
    word_list = list(words)
    word_list.sort(cmp=lambda x, y: frequency[x]-frequency[y])    #Creo que ordena por orden de frecuencia de cada palabra
    #total = reduce(lambda x, k: x + frequency[k], frequency, 0)
        
    min_index = functools.reduce(lambda x, w: x + (1 if frequency[w]/total < min_value else 0), word_list, 0)
    max_index = functools.reduce(lambda x, w: x + (1 if frequency[w]/total < max_value else 0), word_list, 1)
    assert min_index < max_index, "Index errors"
    return word_list[min_index:max_index]

    

def mapping(words, start_at=1):
    if isinstance(words, set):
        words = list(words)
    assert isinstance(words, list), "words should be instance of list"
    to_index = defaultdict(lambda: 0, {words[i]: i+start_at for i in range(0, len(words))})
    from_index = defaultdict(lambda: 'UNKNOWN', {v: k for k, v in to_index.iteritems()})
    return to_index, from_index


if __name__ == '__main__':
    X, y = readFile('D:\\Downloads\\sentiment\\all.review')
    print ('read')
    print (X[0])
    print (y[0])
    X = map(text_to_word_sequence, X)
    print ('words')
    print (X[0])
    print (y[0])
    v, f = vocabulary(X)
    print ('voc')
    print (len(v))
    v = filter_vocabulary(v, f, total=len(X), min_value=0.005, max_value=0.8)
    print ('filter')
    print (len(v))
    to, fr = mapping(v)
    X = [[to[w] for w in s] for s in X]
    X = [filter(lambda x: x > 0, s) for s in X]
    print ('to')
    print (X[0])
    print (y[0])
    X = pad_sequences(X, maxlen=40)
    print ('pading')
    print (X[0])
    print (y[0])
    print ([fr[w] for w in X[0]])

    import cPickle 
    pickle = cPickle
    
    pickle.dump(X, open('X.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(y, open('y.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict(fr), open('fr.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(dict(to), open('to.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(v, open('v.p', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
