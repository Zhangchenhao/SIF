import sys
import os
import gensim

sys.path.append('../src')
import data_io, params, SIF_embedding

import numpy as np


srcWordFilePath = '/opt/zch/iden/vectors-id.txt'
tgtWordFilePath = '/opt/zch/iden/vectors-en.txt'
srcSentFilePath = '/opt/zch/iden/enid.id.txt'
tgtSentFilePath = '/opt/zch/iden/enid.en.txt'
srcsent = open(srcSentFilePath,'r').readlines()
tgtsent = open(tgtSentFilePath,'r').readlines()


# srcsent = ['Pada mulanya, waktu Allah mulai menciptakan alam semesta']
# tgtsent = ['God saw the light, and saw that it was good. God divided the light from the darkness.']
# params = params.params()
weightpara = 1e-3
rmpc = 1

# def srcEmbedding(srcWordFilePath, srcsent):
src_model_300 = gensim.models.KeyedVectors.load_word2vec_format(srcWordFilePath, binary=False)
srcwords = {}
for index, word in enumerate(src_model_300.wv.index2entity):
    srcwords[word] = index
srcWe = src_model_300.wv.vectors
srcword2weight = data_io.getWordWeight(src_model_300.wv.vocab, weightpara)
srcweight4ind = data_io.getWeight(srcwords, srcword2weight)
srcx, srcm = data_io.sentences2idx(srcsent, srcwords)
srcw = data_io.seq2weight(srcx, srcm, srcweight4ind)
srcparams = params.params()
srcparams.rmpc = rmpc
srcEmbedding = SIF_embedding.SIF_embedding(srcWe, srcx, srcw, srcparams)
# return embedding


# def tgtEmbedding(tgtWordFilePath, tgtsent):
tgtmodel_300 = gensim.models.KeyedVectors.load_word2vec_format(tgtWordFilePath, binary=False)
tgtwords = {}
for index, word in enumerate(tgtmodel_300.wv.index2entity):
    tgtwords[word] = index
tgtWe = tgtmodel_300.wv.vectors
tgtword2weight = data_io.getWordWeight(tgtmodel_300.wv.vocab, weightpara)
tgtweight4ind = data_io.getWeight(tgtwords, tgtword2weight)
tgtx, tgtm = data_io.sentences2idx(tgtsent, tgtwords)
tgtw = data_io.seq2weight(tgtx, tgtm, tgtweight4ind)
tgtparams = params.params()
tgtparams.rmpc = rmpc
tgtEmbedding = SIF_embedding.SIF_embedding(tgtWe, tgtx, tgtw, tgtparams)





# srcEmbedding(srcWordFilePath, srcsent)
# tgtEmbedding(tgtWordFilePath, tgtsent)
