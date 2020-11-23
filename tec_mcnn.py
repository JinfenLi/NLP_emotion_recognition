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
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,classification_report
from keras.layers import Input, Dense, Flatten, Dropout, Embedding, GlobalMaxPooling1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate

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
        print(classification_report(val_targ, val_predict,target_names=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'],digits=4),file=open('tec_log_mcnn.log','a'))



        print("-acc: % f — val_f1: % f — val_precision: % f — val_recall % f" % (acc,_val_f1, _val_precision, _val_recall))

        return



emotion_categories = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
# emotion_categories = ['anger', 'fear', 'joy', 'sadness']
num_categories = len(emotion_categories)


# Lexical feature dimension
feature_dimension = 29

filters = [128, 128, 128, 128]
dropout_rates = [0.5, 0.5, 0.5, 0.5]
kernel_sizes = [1, 2, 3, 1]
hidden = [200, 100, 10]

epochs = 5
batch_size = 64
seed = 60

print("processing data")
# X,Y, tweet_matrix, vocab_size, dimension, max_tweet_length,lb,tokenizer_tweets,embeddings_index,\
#            encoded_hash_emo, hash_emo_matrix, max_hash_emo_length,vocab_size_hash_emo,tokenizer_hash_emo,features = tec_process_data.clean_data()
with open('./models/cleandata/tec_cleandata_var.p', 'rb') as f:
    X, Y, tweet_matrix, vocab_size, dimension, max_tweet_length, lb, tokenizer_tweets, embeddings_index, \
    encoded_hash_emo, hash_emo_matrix, max_hash_emo_length, vocab_size_hash_emo,tokenizer_hash_emo, features = pickle.load(f)
# #
# X =X[:600]
# Y = Y[:600]


def model(max_tweet_length, max_hash_emo_length, vocab_size, vocab_size_hash_emo, tweet_matrix, hash_emo_matrix,
          dimension, feature_dimension, num_categories, train_embedding=False):
    # Channel 1
    inputs1 = Input(shape=(max_tweet_length,))
    embedding1 = Embedding(vocab_size, dimension, weights=[tweet_matrix], trainable=train_embedding)(inputs1)
    # 128,1
    # kernel_shape = self.kernel_size + (input_dim, self.filters)
    # 又因为以上的inputdim是最后一维大小(Conv1D中为300，Conv2D中为1），
    # filter数目我们假设二者都是64个卷积核。因此，Conv1D的kernel的shape实际为：
    # （3, 300, 64）
    # 我们假设一个序列是600个单词，每个单词的词向量是300维，那么一个序列输入到网络中就是（600, 300），
    # 当我使用Conv1D进行卷积的时候，实际上就完成了直接在序列上的卷积，卷积的时候实际是以（3, 300）进行卷积，
    # 又因为每一行都是一个词向量，因此使用Conv1D（kernel_size = 3）也就相当于使用神经网络进行了n_gram = 3
    # 的特征提取了。
    conv1 = Conv1D(filters=filters[0], kernel_size=kernel_sizes[0], activation='relu')(embedding1)
    drop1 = Dropout(dropout_rates[0])(conv1)
    pool1 = GlobalMaxPooling1D()(drop1)

    conv2 = Conv1D(filters=filters[1], kernel_size=kernel_sizes[1], activation='relu')(embedding1)
    drop2 = Dropout(dropout_rates[1])(conv2)
    pool2 = GlobalMaxPooling1D()(drop2)

    conv3 = Conv1D(filters=filters[2], kernel_size=kernel_sizes[2], activation='relu')(embedding1)
    drop3 = Dropout(dropout_rates[2])(conv3)
    pool3 = GlobalMaxPooling1D()(drop3)

    # Channel 2
    inputs2 = Input(shape=(max_hash_emo_length,))
    embedding2 = Embedding(vocab_size_hash_emo, dimension, weights=[hash_emo_matrix], trainable=train_embedding)(
        inputs2)
    conv4 = Conv1D(filters=filters[3], kernel_size=kernel_sizes[3], activation='relu')(embedding2)
    drop4 = Dropout(dropout_rates[3])(conv4)
    pool4 = GlobalMaxPooling1D()(drop4)

    # Lexical features
    features = Input(shape=(feature_dimension,))

    merged = concatenate([pool1, pool2, pool3, pool4, features])
    dense1 = Dense(hidden[0], activation='relu')(merged)
    dense2 = Dense(hidden[1], activation='relu')(dense1)
    dense3 = Dense(hidden[2], activation='relu')(dense2)
    outputs = Dense(num_categories, activation='softmax')(dense3)

    model = Model(inputs=[inputs1, inputs2, features], outputs=outputs)
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
        model_GloVe = model(max_tweet_length, max_hash_emo_length, vocab_size, vocab_size_hash_emo, tweet_matrix, hash_emo_matrix,
              dimension, feature_dimension, num_categories, train_embedding=True)
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
        print(classification_report(val_targ, val_predict,
                                    target_names=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'], digits=4),file=open('tec_log_mcnn.log','a'))
        # if counter == 2:
        #     break
    model_GloVe.save("./models/tec_mcnn.h5")
    model_GloVe.save_weights('./models/tec_weight_mcnn.h5')
    import pickle

    pickle.dump((lb, tokenizer_tweets, max_tweet_length, tokenizer_hash_emo, max_hash_emo_length, embeddings_index),
                open("./models/tec_mcnn_var.p", "wb"))
    print("ten folds' average accuracy ", np.mean(accuracies))
    print("ten folds' average f1 ", np.mean(f1))
    print("ten folds' average precision ", np.mean(precision))
    print("ten folds' average recall ", np.mean(recall))
# train_val()
def train_val2():
    model_GloVe = model(max_tweet_length, max_hash_emo_length, vocab_size, vocab_size_hash_emo, tweet_matrix,
                        hash_emo_matrix,
                        dimension, feature_dimension, num_categories, train_embedding=True)
    model_GloVe.fit(x=[X, encoded_hash_emo, features],
                    y=array(Y),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1)

    model_GloVe.save("./models/tec_mcnn_six.h5")
    model_GloVe.save_weights('./models/tec_weight_mcnn_six.h5')
    import pickle

    pickle.dump((lb, tokenizer_tweets, max_tweet_length, tokenizer_hash_emo, max_hash_emo_length, embeddings_index),
                open("./models/tec_mcnn_var_six.p", "wb"))


# train_val2()

def predict():


    # model_GloVe = model(max_tweet_length, max_hash_emo_length, vocab_size, vocab_size_hash_emo, tweet_matrix,
    #                       hash_emo_matrix,
    #                       dimension, feature_dimension, num_categories, train_embedding=True)
    #
    # model_GloVe.load_weights('./models/se_weight_mcnn2.h5', by_name=True)
    model_GloVe = load_model('./models/tec_mcnn_six.h5')

    with open('./models/cleandata/tec_noemotion_cleandata_var.p', 'rb') as f:
        X2, Y2, tweet_matrix2, vocab_size2, dimension2, max_tweet_length2, lb2, tokenizer_tweets2, embeddings_index2, \
        encoded_hash_emo2, hash_emo_matrix2, max_hash_emo_length2, vocab_size_hash_emo2, tokenizer_hash_emo2, features2 = pickle.load(
            f)
    print(max_tweet_length2)
    val_predict = np.argmax(model_GloVe.predict([X2, encoded_hash_emo2, features2]), axis=1)
    val_targ = np.argmax(Y2, axis=1)
    print(accuracy_score(val_targ,val_predict))
    print(classification_report(val_targ, val_predict,
                                 digits=5))
# train_val()
predict()




