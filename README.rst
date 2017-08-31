You can also use the API directly. The following shows basic usage in the
situation where test data is not available at training time::

    import sdm

    # train_features is a list of row-instance data matrices
    # train_labels is a numpy vector of integer categories

    # PCA and standardize the features
    train_feats = sdm.Features(train_features)
    pca = train_feats.pca(varfrac=0.7, ret_pca=True, inplace=True)
    scaler = train_feats.standardize(ret_scaler=True, inplace=True)

    clf = sdm.SDC()
    clf.fit(train_feats, train_labels)
    # ^ gets divergences and does parameter tuning. See the docstrings for
    # more information about options, divergence caches, etc. Caching
    # divergences is highly recommended.

    # get test_features: another list of row-instance data matrices
    # and then process them consistently with the training samples
    test_feats = sdm.Features(test_features, default_category='test')
    test_feats.pca(pca=pca, inplace=True)
    test_feats.normalize(scaler=scaler, inplace=True)

    # get test predictions
    preds = clf.predict(test_feats)

    accuracy = np.mean(preds == test_labels)

To do regression, use ``clf = sdm.NuSDR()`` and a real-valued train_labels;
the rest of the usage is the same.

If you're running on a nontrivial amount of data, it may be nice to pass
``status_fn=True`` and ``progressbar=True`` to the constructor to get status
information out along the way (like in the CLI).

If test data is available at training time, it's preferable to use
``.transduct()`` instead. There's also a ``.crossvalidate()`` method.
