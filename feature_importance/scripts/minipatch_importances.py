from scipy.stats import rankdata


def default_feature_importance(estimator):
    """
    Get feature importance attribute from estimator

    Parameters
    ----------
    estimator : estimator object
        Trained estimator with feature_importances_ attribute

    Returns
    -------
    importances : ndarray
        Feature importances
    """
    return estimator.feature_importances_


def default_feature_importance_rank(estimator, ascending=False, **kwargs):
    """
    Get ranked feature importance attribute from estimator

    Parameters
    ----------
    estimator : estimator object
        Trained estimator with feature_importances_ attribute
    ascending : bool, default=False
        Whether to rank in ascending order
    kwargs : dict
        Additional keyword arguments to pass to scipy.stats.rankdata

    Returns
    -------
    ranks : ndarray
        Rank of each feature
    """
    if ascending:
        return rankdata(estimator.feature_importances_, **kwargs)
    else:
        return rankdata(-estimator.feature_importances_, **kwargs)


def custom_feature_importance_rank(estimator, fn, ascending=False, **kwargs):
    """
    Get ranked feature importances using custom function from estimator

    Parameters
    ----------
    estimator : estimator object
        Trained estimator
    fn : function
        Function to apply to estimator to get feature importances
    ascending : bool, default=False
        Whether to rank in ascending order
    kwargs : dict
        Additional keyword arguments to pass to scipy.stats.rankdata

    Returns
    -------
    ranks : ndarray
        Rank of each feature
    """
    if ascending:
        return rankdata(fn(estimator), **kwargs)
    else:
        return rankdata(-fn(estimator), **kwargs)