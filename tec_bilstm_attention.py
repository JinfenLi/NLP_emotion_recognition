from __future__ import print_function, division
import operator
import numpy as np
from numpy import array
from sklearn.model_selection import KFold
from keras.layers import Bidirectional
import pickle
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
import attention

from EmotionHelpers2 import *
# import tec_process_data
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.accs = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_epoch_end(self, epoch, logs={}):
        loss, acc = self.model.evaluate(self.validation_data[:3], self.validation_data[3], verbose=0)

        val_predict = np.argmax(self.model.predict(self.validation_data[:3]),axis=1)
        val_targ = np.argmax(self.validation_data[3],axis=1)
        _val_f1 = f1_score(val_targ, val_predict,average='weighted')
        _val_recall = recall_score(val_targ, val_predict,average='weighted')
        _val_precision = precision_score(val_targ, val_predict,average='weighted')
        self.accs.append(acc)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(classification_report(val_targ, val_predict,target_names=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'],digits=4))



        print("-acc: % f — val_f1: % f — val_precision: % f — val_recall % f" % (acc,_val_f1, _val_precision, _val_recall))

        return


emotion_categories = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
# emotion_categories = ['anger', 'fear', 'joy', 'sadness']
num_categories = len(emotion_categories)
batch_size = 128
epochs = 15
feature_dimension = 29
seed = 60

print("processing data")
# X,Y, tweet_matrix, vocab_size, dimension, max_tweet_length,lb,tokenizer_tweets,embeddings_index,\
#            encoded_hash_emo, hash_emo_matrix, max_hash_emo_length,vocab_size_hash_emo,tokenizer_hash_emo,features = tec_process_data.clean_data()
with open('./models/cleandata/tec_cleandata_var.p', 'rb') as f:
    X, Y, tweet_matrix, vocab_size, dimension, max_tweet_length, lb, tokenizer_tweets, embeddings_index, \
    encoded_hash_emo, hash_emo_matrix, max_hash_emo_length, vocab_size_hash_emo,tokenizer_hash_emo, features = pickle.load(f)
#
# X =X[:600]
# Y = Y[:600]


def BidLstm(max_tweet_length, max_hash_emo_length, vocab_size, vocab_size_hash_emo, tweet_matrix, hash_emo_matrix,
          dimension, feature_dimension, num_categories, train_embedding=False):
    input1 = Input(shape=(max_tweet_length, ))
    embedding1 = Embedding(vocab_size, dimension, weights=[tweet_matrix],
                  trainable=False)(input1)
    x1 = Bidirectional(LSTM(300, return_sequences=True, dropout=0.25,
                           recurrent_dropout=0.25))(embedding1)
    # pool1 = GlobalMaxPooling1D()(x1)
    x1 = attention.Attention(max_tweet_length)(x1)
    x1 = Dense(256, activation="relu")(x1)
    x1 = Dropout(0.25)(x1)
    # outputs = Dense(num_categories, activation='softmax')(x1)

    input2 = Input(shape=(max_hash_emo_length,))
    embedding2 = Embedding(vocab_size_hash_emo, dimension, weights=[hash_emo_matrix], trainable=train_embedding)(
        input2)
    x2 = Bidirectional(LSTM(300, return_sequences=True, dropout=0.25,
                            recurrent_dropout=0.25))(embedding2)
    # pool2 = GlobalMaxPooling1D()(x2)
    x2 = attention.Attention(max_hash_emo_length)(x2)
    x2 = Dense(256, activation="relu")(x2)
    x2 = Dropout(0.25)(x2)
    # outputs = Dense(num_categories, activation='softmax')(x1)

    # Lexical features
    features = Input(shape=(feature_dimension,))

    merged = concatenate([x1, x2, features])

    merged = Dense(256, activation="relu")(merged)
    merged = Dropout(0.25)(merged)
    outputs = Dense(num_categories, activation='softmax')(merged)

    model = Model(inputs=[input1,input2,features], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_val():
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    metrics = Metrics()
    accuracies = []
    f1=[]
    precision = []
    recall = []
    counter = 1
    model_GloVe = ''
    for train, test in kf.split(X):
        print('Fold#', counter)
        counter += 1
        model_GloVe = BidLstm(max_tweet_length, max_hash_emo_length, vocab_size, vocab_size_hash_emo, tweet_matrix, hash_emo_matrix,
              dimension, feature_dimension, num_categories, train_embedding=True)
        # X_val = X[test][:3]
        # print(X_val)
        # print(len(X_val))
        # y_val = Y[test][:3]
        # print(y_val)
        # model_GloVe.load_weights('./models/tec_weight_biattention.h5', by_name=True)
        model_GloVe.fit(x=[X[train], encoded_hash_emo[train], features[train]],
                        y=array(Y[train]),validation_data=([X[test], encoded_hash_emo[test], features[test]], Y[test]),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[metrics],
                        verbose=1)
        index, value = max(enumerate(metrics.accs), key=operator.itemgetter(1))
        max_f1_index = int(np.argmax(metrics.val_f1s))
        accuracies.append(value)
        f1.append(metrics.val_f1s[max_f1_index])
        precision.append(metrics.val_precisions[max_f1_index])
        recall.append(metrics.val_recalls[max_f1_index])
        print("each epoch's accuracy ",metrics.accs)
        print("each epoch's f1 score ",metrics.val_f1s)
        print("each epoch's precisions ",metrics.val_precisions)
        print("each epoch's recalls ",metrics.val_recalls)
        print("\n")
        val_predict = np.argmax(model_GloVe.predict([X[test], encoded_hash_emo[test], features[test]]), axis=1)
        val_targ = np.argmax(Y[test], axis=1)
        # print(classification_report(val_targ, val_predict,
        #                             target_names=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'], digits=4),file=open('tec_bilstm_log.log','a'))
        # if counter == 2:
        #     break
    model_GloVe.save("./models/tec_biattention.h5")
    model_GloVe.save_weights('./models/tec_weight_biattention.h5')
    import pickle

    pickle.dump((lb, tokenizer_tweets, max_tweet_length, tokenizer_hash_emo, max_hash_emo_length, embeddings_index),
                open("./models/tec_biattention_var.p", "wb"))
    print("ten folds' average accuracy ", np.mean(accuracies))
    print("ten folds' average f1 ", np.mean(f1))
    print("ten folds' average precision ", np.mean(precision))
    print("ten folds' average recall ", np.mean(recall))


def predict():
    model_GloVe = BidLstm(max_tweet_length, max_hash_emo_length, vocab_size, vocab_size_hash_emo, tweet_matrix,
                          hash_emo_matrix,
                          dimension, feature_dimension, num_categories, train_embedding=True)

    model_GloVe.load_weights('./models/tec_weight_bilstm2.h5', by_name=True)
    with open('./models/cleandata/tec_cbet_test_cleandata_var.p', 'rb') as f:
        X2, Y2, tweet_matrix2, vocab_size2, dimension2, max_tweet_length2, lb2, tokenizer_tweets2, embeddings_index2, \
        encoded_hash_emo2, hash_emo_matrix2, max_hash_emo_length2, vocab_size_hash_emo2, tokenizer_hash_emo2, features2 = pickle.load(
            f)
    print(max_tweet_length2)
    val_predict = np.argmax(model_GloVe.predict([X2, encoded_hash_emo2, features2]), axis=1)
    val_targ = np.argmax(Y2, axis=1)
    print(accuracy_score(val_targ, val_predict))
    print(classification_report(val_targ, val_predict,
                                target_names=['anger','disgust', 'fear', 'joy', 'sadness','surprise'], digits=4))
# predict()
# train_val()
def train_val2():
    model_GloVe = BidLstm(max_tweet_length, max_hash_emo_length, vocab_size, vocab_size_hash_emo, tweet_matrix,
                        hash_emo_matrix,
                        dimension, feature_dimension, num_categories, train_embedding=False)
    model_GloVe.fit(x=[X, encoded_hash_emo, features],
                    y=array(Y),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1)

    model_GloVe.save("./models/tec_bilstm_six.h5")
    model_GloVe.save_weights('./models/tec_weight_bilstm_six.h5')
    import pickle

    pickle.dump((lb, tokenizer_tweets, max_tweet_length, tokenizer_hash_emo, max_hash_emo_length, embeddings_index),
                open("./models/tec_bilstm_var_six.p", "wb"))


def blog_tec_compare():
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    metrics = Metrics()
    model_GloVe = BidLstm(max_tweet_length, max_hash_emo_length, vocab_size, vocab_size_hash_emo, tweet_matrix,
                          hash_emo_matrix,
                          dimension, feature_dimension, num_categories, train_embedding=True)
    #
    # model_GloVe.load_weights('./models/tec_weight_bilstm_six.h5', by_name=True)
    for train, test in kf.split(X):
        model_GloVe.fit(x=[X[train], encoded_hash_emo[train], features[train]],
                        y=array(Y[train]), validation_data=([X[test], encoded_hash_emo[test], features[test]], Y[test]),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[metrics],
                        verbose=1)
        model_GloVe.save("./models/tec_story_blog_bilstm.h5")
        model_GloVe.save_weights('./models/tec_story_blog_weight_bilstm.h5')
        break
    with open('./models/cleandata/tec_blog_cleandata_var.p', 'rb') as f:
        X2, Y2, tweet_matrix2, vocab_size2, dimension2, max_tweet_length2, lb2, tokenizer_tweets2, embeddings_index2, \
        encoded_hash_emo2, hash_emo_matrix2, max_hash_emo_length2, vocab_size_hash_emo2, tokenizer_hash_emo2, features2 = pickle.load(
            f)

    val_predict = np.argmax(model_GloVe.predict([X2, encoded_hash_emo2, features2]), axis=1)
    val_targ = np.argmax(Y2, axis=1)
    print(accuracy_score(val_targ, val_predict))
    print(classification_report(val_targ, val_predict,
                                target_names=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'], digits=4))
    with open('./models/cleandata/tec_story_cleandata_var.p', 'rb') as f:
        X3, Y3, tweet_matrix3, vocab_size3, dimension3, max_tweet_length3, lb3, tokenizer_tweets3, embeddings_index3, \
        encoded_hash_emo3, hash_emo_matrix3, max_hash_emo_length3, vocab_size_hash_emo3, tokenizer_hash_emo3, features3 = pickle.load(
            f)

    val_predict = np.argmax(model_GloVe.predict([X3, encoded_hash_emo3, features3]), axis=1)
    val_targ = np.argmax(Y3, axis=1)
    print(accuracy_score(val_targ, val_predict))
    print(classification_report(val_targ, val_predict,
                                target_names=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'], digits=4))


blog_tec_compare()