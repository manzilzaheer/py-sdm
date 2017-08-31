API alignment with D3M::

    class SDC:
        __init__(self, # hyperparameters
                 div_func=DEFAULT_DIV_FUNC,
                 K=DEFAULT_K,
                 tuning_folds=DEFAULT_TUNING_FOLDS,
                 n_proc=None,
                 C_vals=DEFAULT_C_VALS,
                 sigma_vals=DEFAULT_SIGMA_VALS, scale_sigma=True,
                 svr_nu_vals=DEFAULT_SVR_NU_VALS,
                 cache_size=DEFAULT_SVM_CACHE,
                 tuning_cache_size=DEFAULT_SVM_CACHE,
                 svm_tol=DEFAULT_SVM_TOL,
                 tuning_svm_tol=DEFAULT_SVM_TOL,
                 svm_max_iter=DEFAULT_SVM_ITER,
                 tuning_svm_max_iter=DEFAULT_SVM_ITER_TUNING,
                 svm_shrinking=DEFAULT_SVM_SHRINKING,
                 status_fn=None, progressbar=None,
                 min_dist=None,
                 symmetrize_divs=DEFAULT_SYMMETRIZE_DIVS,
                 km_method=DEFAULT_KM_METHOD,
                 transform_test=DEFAULT_TRANSFORM_TEST,
                 save_bags=True)):
            return
        
        fit(self, X, y, sample_weight=None, divs=None, divs_cache=None, ret_km=False): # No classes, not sure what is it!!!
            return
            
        predict(self, X, divs=None, km=None): # Divs and km extra
            return y
            
        predict_log_proba(self, X):             
            raise NotImplementedError
            
        staged_fit(self, X, y, sample_weight=None, classes=None, **kwargs):
            raise NotImplementedError
  
        staged_predict(self, X):
            raise NotImplementedError
            
        staged_predict_log_proba(self, X):
            raise NotImplementedError
            
        __getstate__(self):
            raise NotImplementedError
            
        __setstate__(self, state):
            raise NotImplementedError
    
    
    import sdm

    # train_features is a list of row-instance data matrices
    # train_labels is a numpy vector of integer categories

    clf = sdm.SDC()
    clf.fit(train_features, train_labels)
    # ^ gets divergences and does parameter tuning. See the docstrings for
    # more information about options, divergence caches, etc. Caching
    # divergences is highly recommended.

    # get test_features: another list of row-instance data matrices
    # and then process them consistently with the training samples
    # get test predictions
    preds = clf.predict(test_features)

    accuracy = np.mean(preds == test_labels)

To do regression, use ``clf = sdm.NuSDR()`` and a real-valued train_labels;
the rest of the usage is the same.

If you're running on a nontrivial amount of data, it may be nice to pass
``status_fn=True`` and ``progressbar=True`` to the constructor to get status
information out along the way (like in the CLI).
