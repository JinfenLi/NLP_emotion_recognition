import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,classification_report

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.accs = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_epoch_end(self, epoch, logs={}):
        # print(self.validation_data)
        # print(len(self.validation_data))
        # print(self.validation_data[-1])
        # print(self.model.predict(self.validation_data[:3]))
        # print(np.argmax(self.model.predict(self.validation_data[:3]),axis=1))
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
        # print(classification_report(val_targ, val_predict,target_names=['anger', 'disgust', 'fear', 'guilt', 'joy', 'love', 'sadness', 'surprise', 'thankfulness'],digits=4))
        # print(classification_report(val_targ, val_predict,target_names=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'],digits=4))
        print(classification_report(val_targ, val_predict,target_names=['anger', 'fear', 'joy', 'sadness'],digits=4))



        print("-acc: % f — val_f1: % f — val_precision: % f — val_recall % f" % (acc,_val_f1, _val_precision, _val_recall))

        return

