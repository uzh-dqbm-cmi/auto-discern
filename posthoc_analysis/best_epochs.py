import pickle
import numpy as np


indicies = []
for q in [4, 5, 9, 10, 11]:
    for f in range(5):
        p = './question_{}/fold_{}/score_validation.pkl'.format(q, f)
        with open(p, 'rb+') as f:
            x = pickle.load(f)
        # don't forget to add 1 to that, becuase these are indicies, not number of epochs!

        indicies.append(x.best_epoch_indx+1)
print(indicies)
print(sum(indicies)/len(indicies))
print(np.std(indicies))
