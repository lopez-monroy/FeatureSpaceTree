def f1():
    vocabulary = list(set([ str(random.randint(0,200000)) for e in range(200000)]))
    valid_tokens = [ str(random.randint(0,200000)) for e in range(6000)]

    t1 = time.time()
    local_vocabulary = set(vocabulary) & set(valid_tokens)
    valid_tokens = [token for token in valid_tokens if token in local_vocabulary]
    print "vocabulary: " + str(len(vocabulary))
    print "valid tokens: " + str(len(valid_tokens))
    print "local vocabulary: " + str(len(local_vocabulary))
    print "time: ", time.time()-t1

def f2():
    vocabulary = {str(random.randint(0,200000)):random.randint(0,20000) for e in range(200000)}
    valid_tokens = [ str(random.randint(0,200000)) for e in range(6000)]

    t1 = time.time()
    valid_tokens = [token for token in valid_tokens if token in vocabulary]
    print "vocabulary: " + str(len(vocabulary))
    print "valid tokens: " + str(len(valid_tokens))
    print "time: ", time.time()-t1

def f3():
    vocabulary = set([ str(random.randint(0,200000)) for e in range(200000)])
    valid_tokens = [ str(random.randint(0,200000)) for e in range(6000)]

    t1 = time.time()
    local_vocabulary = vocabulary & set(valid_tokens)
    valid_tokens = [token for token in valid_tokens if token in local_vocabulary]
    print "vocabulary: " + str(len(vocabulary))
    print "valid tokens: " + str(len(valid_tokens))
    print "local vocabulary: " + str(len(local_vocabulary))
    print "time: ", time.time()-t1

def f4():
    vocabulary = nltk.FreqDist([str(random.randint(0,200000)) for e in range(200000)])
    valid_tokens = [ str(random.randint(0,200000)) for e in range(6000)]

    t1 = time.time()
    valid_tokens = [token for token in valid_tokens if token in vocabulary]
    print "vocabulary: " + str(len(vocabulary))
    print "valid tokens: " + str(len(valid_tokens))
    print "time: ", time.time()-t1


def test_vocabs():
    vocabulary = list(set([ str(random.randint(0,200000)) for e in range(200000)]))
    len_vocab = len(vocabulary)
    matrix_docs_terms = numpy.zeros((1000, len_vocab),
                                    dtype=numpy.float64)
    unorder_dict_index = {}
    for (term, u) in zip(vocabulary, range(len_vocab)):
        unorder_dict_index[term] = u

    t1 = time.time()
    i = 0
    for h in range(1000):
        docActualFd = nltk.FreqDist(set([ str(random.randint(0,200000)) for e in range(6000)]) & set(vocabulary))
        for pal in docActualFd:
            if (pal in unorder_dict_index):
                freq = docActualFd[pal] / float(len(docActualFd))
            else:
                freq = 0.0
                        
            matrix_docs_terms[i, unorder_dict_index[pal]] = freq

        i+=1
    print "time: ", time.time()-t1
    return matrix_docs_terms

def test_vocabs_SKIP():
    vocabulary = list(set([ str(random.randint(0,200000)) for e in range(200000)]))
    len_vocab = len(vocabulary)
    matrix_docs_terms = numpy.zeros((1000, len_vocab),
                                    dtype=numpy.float64)
    unorder_dict_index = {}
    for (term, u) in zip(vocabulary, range(len_vocab)):
        unorder_dict_index[term] = u

    t1 = time.time()
    i = 0
    for h in range(1000):
        docActualFd = nltk.FreqDist(set([ str(random.randint(0,200000)) for e in range(6000)]) & set(vocabulary))
        for pal in docActualFd:
            freq = 0.0
            freq = docActualFd[pal] / float(len(docActualFd))                       
            matrix_docs_terms[i, unorder_dict_index[pal]] = freq

        i+=1
    print "time: ", time.time()-t1
    return matrix_docs_terms


def test_vocabs_L():
    
    vocabulary = list(set([ str(random.randint(0,200000)) for e in range(200000)]))
    len_vocab = len(vocabulary)
    matrix_docs_terms = numpy.zeros((1000, len_vocab),
                                    dtype=numpy.float64)
    unorder_dict_index = {}
    for (term, u) in zip(vocabulary, range(len_vocab)):
        unorder_dict_index[term] = u

    t1 = time.time()
    i = 0
    for h in range(1000):
        docActualFd = nltk.FreqDist(set([ str(random.randint(0,200000)) for e in range(6000)]) & set(vocabulary))
        j=0
        for pal in vocabulary:
            
            if pal in docActualFd:
                freq = docActualFd[pal] / float(len(docActualFd))
            else:
                freq = 0.0
                            
            matrix_docs_terms[i, j] = freq

            j += 1

        i+=1
    print "time: ", time.time()-t1
    return matrix_docs_terms
                            

    


