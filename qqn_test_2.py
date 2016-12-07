import numpy 
import cPickle 
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Sequential


if __name__ == '__main__':
    
    pickle = cPickle
    X = pickle.load(open('X.p', mode='rb'))
    y = pickle.load(open('y.p', mode='rb'))
    v = pickle.load(open('v.p', mode='rb'))
    model = Sequential()
    model.add(Embedding(len(v)+1, 100))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    s = int(len(X)*.7)
    
    np = numpy
    
    X_train = np.asarray(X[:s])
    y_train = np.asarray(y[:s])
    X_test = np.asarray(X[s:])
    y_test = np.asarray(y[s:])
    for i in range(0, 1000):
        print ('Epoch {}'.format(i))
        model.fit(X_train, y_train, nb_epoch=1, batch_size=500)
        print (model.evaluate(X_test, y_test, batch_size=10000))