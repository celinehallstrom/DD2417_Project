import pandas as pd
import numpy as np
from tqdm import tqdm 
import spacy
import nltk
#nltk.download('stopwords')
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def clean_line(line):
  line_list = line.split()
  cleaned_line = []
  for word in line_list:
      word = ''.join([ch.lower() for ch in list(word) if ch.isalpha()])
      if word != "":
          cleaned_line.append(word)
  return cleaned_line

def remove_stopwords(st, stop_words):
  str = ' '.join([w for w in st if not w.lower() in stop_words])
  return str

def cosine_sim(q1_terms_nsw, q2_terms_nsw, model):
  sentence1 = " ".join([str(item) for item in q1_terms_nsw])
  sentence2 = " ".join([str(item) for item in q2_terms_nsw])
  sentence_vecs1 = model.encode(sentence1)
  sentence_vecs2 = model.encode(sentence2)
  cs = cosine_similarity([sentence_vecs1], sentence_vecs2.reshape(1, -1))
  return cs[0][0]

def pos_neg(q_terms, df_pos_neg):
  pos = 0
  neg = 0
  for elem in q_terms:
    if elem in df_pos_neg['Positive'].values:
      pos += 1
    if elem in df_pos_neg['Negative'].values:
      neg += 1  
  return pos, neg

def features(q1_terms, q2_terms, df_pos_neg):
  num_words_q1 = len(q1_terms)
  num_words_q2 = len(q2_terms)
  if q1_terms[-1] == q2_terms[-1]:
    same_end = 1
  else:
    same_end = 0
  q1_pos, q1_neg = pos_neg(q1_terms, df_pos_neg)
  q2_pos, q2_neg = pos_neg(q2_terms, df_pos_neg)

  return [num_words_q1, num_words_q2, q1_pos, q1_neg, q2_pos, q2_neg, same_end]

def features_nsw(q1_nsw, q2_nsw, model):
  q1_terms_nsw = q1_nsw.split(' ')
  q2_terms_nsw = q2_nsw.split(' ')
  cosine = cosine_sim(q1_terms_nsw, q2_terms_nsw, model)
  num_similar = 0
  wv1 = create_w2v(q1_nsw)
  wv2 = create_w2v(q2_nsw)
  wv_dot = np.dot(wv1, wv2)/(np.linalg.norm(wv1)*np.linalg.norm(wv2))
  for term in q1_terms_nsw:
    if term in q2_terms_nsw:
      num_similar += 1
  return [cosine, num_similar, wv_dot]

def create_feature_vectors(df):
  model_name = 'bert-base-nli-mean-tokens'
  model = SentenceTransformer(model_name) 
  feature_names = ['id', 'num_words_q1', 'num_words_q2', 'q1_pos', 'q1_neg', 'q2_pos', 'q2_neg', 
                  'same_end', 'cosine', 'num_similar', 'wv_dot', 'is_duplicate']

  stop_words = set(nltk.corpus.stopwords.words('english'))
  df_pos_neg = pd.read_excel('data/Positive and Negative Word List.ods', engine='odf')
  df_fv = pd.DataFrame(columns=feature_names)
  for i in tqdm(range(len(df))): 
      id = df.iloc[i]['id'] 
      q1 = df.iloc[i]['question1']
      q2 = df.iloc[i]['question2']
      try:
        q1_terms = clean_line(q1)
        q2_terms = clean_line(q2)
      except Exception:
        print('Could not clean line', i)
      if len(q1_terms) >= 1 and len(q2_terms) >= 1:
        try:
          q1_nsw = remove_stopwords(q1_terms, stop_words)
          q2_nsw = remove_stopwords(q2_terms, stop_words)
        except Exception:
          print('Could not remove stopwords on line', i)
        try:
          fv1 = features(q1_terms, q2_terms, df_pos_neg)
          fv2 = features_nsw(q1_nsw, q2_nsw, model)
          fv = fv1 + fv2
          fv.insert(0, id)
          fv.append(df.iloc[i]['is_duplicate'])
          df_fv.loc[len(df_fv)] = fv
        except Exception:
          print('Could not create features on line', i)
        
        if i == 0:
            df_fv.to_csv('all_features.csv', index_label='datapoint')
            df_fv = pd.DataFrame(columns=feature_names)
        elif i % 10000 == 0:
            df_fv.to_csv('all_features.csv', index_label='datapoint', header=None, mode='a')
            df_fv = pd.DataFrame(columns=feature_names)
  df_fv.to_csv('all_features.csv', index_label='datapoint', header=None, mode='a')

def create_w2v(question):
  nlp = spacy.load("en_core_web_sm")
  doc = nlp(question)
  for t in range(len(doc)):
      if t == 0:
          wv = doc[t].vector
      else:
          wv += doc[t].vector
  return (1/len(doc))*wv

def create_w2v_feature(question_pairs):
    column_names = ['is_duplicate']
    stop_words = set(nltk.corpus.stopwords.words('english'))
    for i in range(1, 97):
      column_names.append('dim' +str(i))
      column_names = ['id', 'is_duplicate', 'w2v_dot']
    df_fv = pd.DataFrame(columns=column_names)
    for i in tqdm(range(len(question_pairs))):
        id = question_pairs.iloc[i]['id']
        
        q1 = question_pairs.iloc[i]['question1']
        q2 = question_pairs.iloc[i]['question2']
        try:
            q1_terms = clean_line(q1)
            q2_terms = clean_line(q2)
            if len(q1_terms) >= 1 and len(q2_terms) >= 1:
                q1_nsw = remove_stopwords(q1_terms, stop_words)
                q2_nsw = remove_stopwords(q2_terms, stop_words)
                wv1 = create_w2v(q1_nsw)
                wv2 = create_w2v(q2_nsw)
                wv_dot = np.dot(wv1, wv2)/(np.linalg.norm(wv1)*np.linalg.norm(wv2))
                wv = [id, question_pairs.iloc[i]['is_duplicate'], wv_dot]
                df_fv.loc[len(df_fv)] = wv
        except Exception:
            print("Could not save id: ", id)     
        if i == 0:
            print("save")
            df_fv.to_csv('w2v.csv', index_label='datapoint')
            df_fv = pd.DataFrame(columns=column_names)
        elif i % 1000 == 0:
            print("save")
            df_fv.to_csv('w2v.csv', header=None, index_label='datapoint', mode='a')
            df_fv = pd.DataFrame(columns=column_names) 
            #df.to_csv('labels.csv', header=None, mode='a')
            #df = pd.DataFrame(columns=['id', 'is_duplicate'])
    df_fv.to_csv('w2v.csv', header=None, index_label='datapoint', mode='a')

def create_final_df():
    column_names =['id', 'num_words_q1', 'num_words_q2', 'cosine', 'num_similar', 'q1_pos', 'q1_neg', 'q2_pos', 'q2_neg', 'same_end', 'is_duplicate']
    df = pd.read_csv('data/feature_vectors.csv', usecols=column_names).dropna()
    df.drop(df[df['is_duplicate'] > 1].index, inplace = True)
    df.reset_index(drop=True, inplace=True)
    
    column_names = ['id', 'w2v_dot']
    df_wv = pd.read_csv('data/w2v_1.csv', usecols=column_names, dtype=float)
    df_wv.reset_index(drop=True, inplace=True)
    df_new = pd.merge(df, df_wv, on='id', how='outer').drop_duplicates().fillna(0)
    labels = df_new['is_duplicate']
    df_features = df_new.drop(columns=['is_duplicate', 'id'])
    return df_new, df_features, labels

#df = pd.read_csv('data/question_pairs_final.csv', usecols=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'], dtype='unicode')

# THERE ARE DUPLICATE ID:S IN ORIGINAL DATASET
# Run to create original feature vectors, takes ~ 16 h
#create_feature_vectors(df)

# Run to create w2v cosine feature
#create_w2v_feature(df) 



