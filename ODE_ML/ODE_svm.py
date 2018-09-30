from utils import *
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

print('\nStart')
n = 8 #dimension of systems
cut = 0.5 #procent of stable samples
N = 10000 #number of samples
d = 10 #~dispersion

if sys.argv[1].lower()=='create':
	tt = time()
	X_train, Y_train = generate_samples(n, cut, N, d, flag=True)
	X_test, Y_test = generate_samples(n, cut, N//5, d, flag=True)
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
clf = SVC(verbose=True)
clf.fit(StandardScaler().fit_transform(X_train), Y_train)



Y_pred = (clf.predict(StandardScaler().fit_transform(X_test))>0.5).astype('int')
acc_score = accuracy_score(Y_test, Y_pred)
roc_score = roc_auc_score(Y_test, Y_pred)
cm = confusion_matrix(Y_test, Y_pred)
print('MEAN pred:', np.mean(Y_pred))
print('ACC:', acc_score)
print('ROC_AUC:', roc_score)
print('CONFUSION MATRIX\n', cm / cm.sum(axis=1))

print('Done\n\n')





