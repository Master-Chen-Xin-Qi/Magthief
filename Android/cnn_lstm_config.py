import pickle
config = {'train_batch_size': 800,
         'MAX_NB_VARIABLES': 100,
         'batch_size': 20}

selfref_list = [1, 2, 3]
selfref_list.append(selfref_list)

output = open('data.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(config, output)

# Pickle the list using the highest protocol available.
pickle.dump(selfref_list, output, -1)

output.close()
