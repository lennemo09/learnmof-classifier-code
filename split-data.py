import splitfolders
splitfolders.ratio("dataset_3classes_corrected", output="dataset_split-70-15-15",
                   seed=1337, ratio=(.7, .15, .15), group_prefix=None, move=False)