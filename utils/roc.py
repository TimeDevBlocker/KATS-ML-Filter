from sklearn import metrics
from read_data import *
import matplotlib.pylab as plt
import openpyxl
plt.rc('font', family='Arial')

filename = 'combined.xlsx'

y_true_1 = read_first_column(filename)

y_score_2 = read_second_column(filename)

fpr1, tpr1, thresholds = metrics.roc_curve(y_true_1, y_score_2)
roc_auc1 = metrics.auc(fpr1, tpr1)  # the value of roc_auc1
print(roc_auc1)
plt.plot(fpr1, tpr1, 'b', label='AUC = %0.5f' % roc_auc1)

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0, 1])  # the range of x-axis
# plt.ylim([0, 1])  # the range of y-axis
plt.xlabel('False Positive Rate')  # the name of x-axis
plt.ylabel('True Positive Rate')  # the name of y-axis
plt.title('ROC Curve for KATS Real/Bogus Classifier')  # the title of figure
plt.savefig("demo.jpg",dpi=1200)
plt.show()
