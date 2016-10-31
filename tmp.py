import tools

input_dict, symbol_dict = tools.build_data_pickles()
X, y = tools.dict_to_list(input_dict,symbol_dict)
clf = train_mlp(X, y, verbose=True)
print(clf.score)
