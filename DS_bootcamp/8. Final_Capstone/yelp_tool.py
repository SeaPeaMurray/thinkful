import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
import spacy
import yelp_tool
from spacy_readability import Readability
from gensim.models.doc2vec import TaggedDocument
from gensim.corpora import Dictionary
from gensim.similarities import Similarity
from gensim.models import LsiModel
from gensim.test.utils import get_tmpfile
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from joblib import dump, load

# Basic text cleaning
def fix_nl(mytext):
    text = re.sub(r'\n\n', '', mytext)
    text = re.sub(r'\n', '', text)
    final = re.sub(r'  ', ' ', text)
    return final

# Basic lemmatize function
def lemma_sent(sent):
    return ' '.join(word.lemma_.lower() for word in sent if not word.is_punct and not word.is_stop)

# Preprocessor
def preproc(test, n, nlp, noval_index, val_index, mydct, pca8):
    test_sub = test.sample(n=n)
    X1, X2, y = [], [], []
    # Preprocess readability and length
    doc = nlp.pipe(test_sub.text)
    read_len = [(txt._.flesch_kincaid_reading_ease, len(txt)) for txt in doc]
    # Preprocess embeddings
    test_sub.text = test_sub.text.apply(fix_nl)
    for row in test_sub.itertuples():
        X1.append(noval_index[mydct.doc2bow(fix_nl(row.text).split())])
        X2.append(val_index[mydct.doc2bow(fix_nl(row.text).split())])
        y.append(row.useful)
    # PCA8 transform **ONLY**
    Xa = np.concatenate([X1, X2], axis=1)[:,:900000] # Restrict size with index
    pca8_test_X = pca8.transform(Xa)
    return pca8_test_X, y, read_len
    
# Modeler
def modeler(X, y, read_len, vec_model, rnl_model):
    # TSNE for graphing
    tsne = TSNE()
    graph = tsne.fit_transform(X)
    test_y = np.array(y)
    # Dicationary values
    vec_proba = vec_model.predict_proba(X=X)
    rnl_proba = rnl_model.predict_proba(np.array(read_len))
    comb_proba = rnl_proba + vec_proba
    pred = None
    success = None
    # Return variables
    score_and_y = {'proba':comb_proba, 'y':y, 'comb_pred':pred, 'success':success, 'graph':graph}
    score_and_y['pred'] = score_and_y['proba'][:,1] > score_and_y['proba'][:,0]
    score_and_y['success'] = score_and_y['y'] == score_and_y['pred']
    return score_and_y

# Load variables
def vars():
	mydct = load('mydct.joblib')
	noval_corp = load('noval_corp.joblib')
	noval_ind = get_tmpfile('index')
	noval_index = Similarity(noval_ind, noval_corp, len(mydct))
	val_corp = load('val_corp.joblib')
	val_ind = get_tmpfile('index')
	val_index = Similarity(val_ind, val_corp, len(mydct))
	pca8 = load('pca8.joblib')
	nlp = yelp_tool.spacy.load('en_core_web_md', disable=['tagger', 'ner'])
	read = yelp_tool.Readability()
	nlp.add_pipe(read, last=True)
	return mydct, noval_index, val_index, pca8, nlp

