from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix


y_target =    [1, 1, 1, 0, 0, 2, 0, 3]
y_predicted = [1, 0, 1, 0, 0, 2, 1, 3]

cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted, 
                      binary=False)

import matplotlib.pyplot as plt

fig, ax = plot_confusion_matrix(conf_mat=cm)
plt.show()
plt.savefig("test_save.png")
