import pickle, sys
sys.path.append('../src')
import data_io, sim_algo, eval, params
import gensim
## run
# wordfiles = [#'../data/paragram_sl999_small.txt', # need to download it from John Wieting's github (https://github.com/jwieting/iclr2016)
#     '../data/vectors-en.txt'  # need to download it first
#     ]
# weightfile = '../auxiliary_data/enwiki_vocab_min200.txt'
# weightparas = [-1, 1e-3]#[-1,1e-1,1e-2,1e-3,1e-4]
# rmpcs = [0,1]# [0,1,2]
wordFilePath = '../data/vectors-en.txt'
model_300 = gensim.models.KeyedVectors.load_word2vec_format(wordFilePath,binary = False)
words={}
for index,word in enumerate(model_300.wv.index2entity):
    words[word] = index
We = model_300.wv.vectors
weightparas = [-1, 1e-3]
rmpcs = [0,1]

params = params.params()
parr4para = {}
sarr4para = {}

    # (words, We) = data_io.getWordmap(wordfile)
for weightpara in weightparas:
    word2weight = data_io.getWordWeight(model_300.wv.vocab, weightpara)
    weight4ind = data_io.getWeight(words, word2weight)
    for rmpc in rmpcs:
        print ('word vectors loaded from %s' % wordFilePath)
        # print ('word weights computed from %s using parameter a=%f' % (weightfile, weightpara))
        params.rmpc = rmpc
        print ('remove the first %d principal components' % rmpc)
        ## eval just one example dataset
        parr, sarr = eval.sim_evaluate_one(We, words, weight4ind, sim_algo.weighted_average_sim_rmpc, params)
        ## eval all datasets; need to obtained datasets from John Wieting (https://github.com/jwieting/iclr2016)
        # parr, sarr = eval.sim_evaluate_all(We, words, weight4ind, sim_algo.weighted_average_sim_rmpc, params)
        paras = (wordFilePath, model_300.wv.vocab, weightpara, rmpc)
        parr4para[paras] = parr
        sarr4para[paras] = sarr

## save results
save_result = False #True
result_file = 'result/sim_sif.result'
comment4para = [ # need to align with the following loop
    ['word vector files', wordFilePath], # comments and values,
    ['weight parameters', weightparas],
    ['remove principal component or not', rmpcs]
]
if save_result:
    with open(result_file, 'w') as f:
        pickle.dump([parr4para, sarr4para, comment4para] , f)

