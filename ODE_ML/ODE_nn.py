from utils import *

import keras
from keras import regularizers
from keras import Sequential
from keras.layers import *
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.models import model_from_json



print('\nStart')
n = 8 #dimension of systems
cut = 0.5 #procent of stable samples
N = 100000 #number of samples
d = 10 #~dispersion

if sys.argv[1].lower()=='create':
	tt = time()
	X_train, Y_train = generate_samples(n, cut, N, d)
	X_test, Y_test = generate_samples(n, cut, N//5, d)
	print('Creating samples, time:', (time()-tt))
	pickle_dump(X_train, 'z_X_train.pckl')
	pickle_dump(Y_train, 'z_Y_train.pckl')
	pickle_dump(X_test, 'z_X_test.pckl')
	pickle_dump(Y_test, 'z_Y_test.pckl')
else:
	X_train = pickle_load('z_X_train.pckl')
	Y_train = pickle_load('z_Y_train.pckl')
	X_test = pickle_load('z_X_test.pckl')
	Y_test = pickle_load('z_Y_test.pckl')
	print('Loading done')


print('Number of features:', np.shape(X_train)[1])

layer_sizes = [64, 32, 16, 8, 4, 2]


model_input = Input(shape=(np.shape(X_train)[1],))
model_layer = Dense(layer_sizes[0], activation='relu')(model_input)
model_layer = Dropout(0.5)(model_layer)
for s in layer_sizes[1:]:
	model_layer = Dense(s, activation='relu')(model_layer)
	model_layer = Dropout(0.5)(model_layer)
model_layer = Dense(1, activation='sigmoid')(model_layer)
target = Dropout(0.5)(model_layer)

model = keras.Model(inputs=model_input, outputs=target)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    


# model.summary()
model.fit(X_train, Y_train, epochs=100, batch_size=N, validation_split=0.1)



Y_pred = (model.predict(X_test).flatten()>0.5).astype('int')
acc_score = accuracy_score(Y_test, Y_pred)
roc_score = roc_auc_score(Y_test, Y_pred)
cm = confusion_matrix(Y_test, Y_pred)

print('MEAN pred:', np.mean(Y_pred))
print('ACC:', acc_score)
print('ROC_AUC:', roc_score)
print('CONFUSION MATRIX\n', cm / cm.sum(axis=1))


print('Done\n\n')





