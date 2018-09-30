import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import codecs

import locale
locale.setlocale(locale.LC_ALL, "deu_deu")
matplotlib.rcParams['axes.formatter.use_locale'] = True



data_dLTI = pd.read_csv('g_dLTI.csv', encoding="utf-8")
h1 = data_dLTI['h'].ravel()
gamma1 = data_dLTI['gamma_dLTI'].ravel()

data_LMI = pd.read_csv('g_LMI.csv', encoding="utf-8")
h2 = data_LMI['h'].ravel()
gamma2 = data_LMI['gamma_LMI'].ravel()



plt.rc('text', usetex=True)
plt.rc('font', family='sans')
gammad, = plt.plot(h1, gamma1, label=r'$\gamma_d$', linewidth=2.0)
lmi, = plt.plot(h2, gamma2, '--', label='LMI')
plt.legend([gammad, lmi], [r'$\gamma_d$', 'LMI'], prop={'size': 20})

plt.tick_params(direction='out')
plt.xlabel(r'$h$', fontsize=22, labelpad=-1)

plt.xticks(fontsize=20)
plt.xticks([0.2, 0.4, 0.6, 0.8, 1])
plt.xlim([0, 1])

plt.yticks(fontsize=20)
plt.ylabel(r"\textbf{$\gamma$}", rotation=0, fontsize=22, labelpad=10)
plt.ylim([1, 6])


plt.show()
# plt.savefig("image.jpg", bbox_inches='tight')




