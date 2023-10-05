import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from scipy.stats import norm, rankdata


class _MinipatchBase(BaseEstimator):
    """
    Base class for minipatch estimators.

    Parameters
    ----------
    estimator : sklearn-compatible object
        The base estimator to fit on minipatches.
    n_ratio : float or str
        The ratio of samples to use in each minipatch. If 'sqrt', use sqrt(n) samples.
    p_ratio : float or str
        The ratio of features to use in each minipatch. If 'sqrt', use sqrt(p) features.
    num_mps : int
        The number of minipatches to use.
    importance_fn : callable, optional
        The function to compute the feature importance scores for each minipatch.
    random_state : not yet implemented
    """
    def __init__(self, estimator, n_ratio, p_ratio, num_mps, 
                 importance_fn=None, random_state=None):
        self.estimator = estimator
        self.n_ratio = n_ratio
        self.p_ratio = p_ratio
        self.num_mps = num_mps
        self.importance_fn = importance_fn
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """
        Fit the minipatch model to the given training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The covariate matrix.
        y : ndarray of shape (n_samples,)
            The observed responses.
        sample_weight : ndarray of shape (n_samples,), optional
            Sample weights. If None, all samples are weighted equally.
        """
        self.estimators_ = []
        self.mp_samples_ = []
        self.mp_features_ = []
        self.predictions_ = []
        self.oob_predictions_ = None
        self.oob_score_ = None
        self.y = y
        self.train_n = X.shape[0]
        self.train_p = X.shape[1]
        # np.random.seed(self.random_state)

        if isinstance(self.estimator, ClassifierMixin):
            self.classes_ = np.unique(y)
            self._pred_fun = self._validate_classes(
                self.estimator.predict_proba(X),
                self.estimator.classes_
            )
        else:
            self._pred_fun = self.estimator.predict

        if sample_weight is None:
            sample_weight = self._default_sample_weight(y)

        # fit estimators on num_mps minipatches
        for k in tqdm(range(self.num_mps)):
            self._fit_mp(X, y, sample_weight)

        # compute OOB predictions
        preds_all = copy.deepcopy(self.predictions_)
        for k, idx_n in enumerate(self.mp_samples_):
            preds_all[k][idx_n] = np.nan
        self.oob_predictions_ = np.nanmean(preds_all, axis=0)
        self.oob_score_ = self._default_score(y, self.oob_predictions_)

    def _fit_mp(self, X, y, sample_weight=None, save_estimator=True):
        """
        Fit an estimator on a single minipatch to the given training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The covariate matrix.
        y : ndarray of shape (n_samples,)
            The observed responses.
        sample_weight : ndarray of shape (n_samples,), optional
            Sample weights. If None, all samples are weighted equally.
        save_estimator : bool, default=True
            Whether to save the estimator.
        """
        idx_n, idx_p = self._get_mp_idxs(sample_weight)
        X_train = X[idx_n, :][:, idx_p]
        y_train = y[idx_n]
        self.estimator.fit(X_train, y_train)
        if save_estimator:
            self.estimators_.append(copy.deepcopy(self.estimator))
        self.mp_samples_.append(idx_n)
        self.mp_features_.append(idx_p)
        preds = self._pred_fun(X[:, idx_p])
        self.predictions_.append(copy.deepcopy(preds))

    def predict(self, X, type='response'):
        """
        Predict on the given test data.

        Parameters
        ----------
        X : ndarray of shape (n_test_samples, n_features)
            The covariate matrix for the test data.
        type : str, one of {'response', 'all'}
            If 'response', return the average of the predictions from each minipatch.
            If 'all', return the predictions from each minipatch.

        Returns
        -------
        predictions : ndarray of shape (n_test_samples,) if type='response', or
            ndarray of shape (n_test_samples, num_mps) if type='all'
            The predictions.
        """
        assert type in ['response', 'all']
        X = check_array(X)
        check_is_fitted(self, "estimators_")
        if type == 'response':
            predictions = 0
            for estimator, idx_p in zip(self.estimators_, self.mp_features_):
                predictions += estimator.predict(X[:, idx_p])
            predictions = predictions / self.num_mps
        elif type == 'all':
            predictions = []
            for estimator, idx_p in zip(self.estimators_, self.mp_features_):
                predictions.append(estimator.predict(X[:, idx_p]))
            predictions = np.array(predictions).T
        return predictions
    
    def fit_predict(self, X, y, X_test, sample_weight=None, type='response'):
        """
        Fit the model to the given training data and predict on the given test data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The covariate matrix.
        y : ndarray of shape (n_samples,)
            The observed responses.
        X_test : ndarray of shape (n_test_samples, n_features)
            The covariate matrix for the test data.
        sample_weight : ndarray of shape (n_samples,), optional
            Sample weights. If None, all samples are weighted equally.
        type : str, one of {'response', 'all'}
            If 'response', return the average of the predictions from each minipatch.
            If 'all', return the predictions from each minipatch.

        Returns
        -------
        test_preds : ndarray of shape (n_test_samples,) if type='response', or
            ndarray of shape (n_test_samples, num_mps) if type='all'
            The predictions.
        test_proba_preds: ndarray of shape (n_test_samples, n_classes) if type='response', or
            num_mps-length list of ndarrays of shape (n_test_samples, n_classes) if type='all'
            The class probability predictions. Only for MinipatchClassifier.
        """
        assert type in ['response', 'all']

        self.mp_samples_ = []
        self.mp_features_ = []
        self.predictions_ = []
        self.oob_predictions_ = None
        self.oob_score_ = None
        self.feature_importances_all_ = None
        self.feature_importances_ = None
        self.y = y
        self.train_n = X.shape[0]
        self.train_p = X.shape[1]
        # np.random.seed(self.random_state)
        
        if isinstance(self.estimator, ClassifierMixin):
            self.classes_ = np.unique(y)
            self._pred_fun = self._validate_classes(
                self.estimator.predict_proba(X),
                self.estimator.classes_
            )
        else:
            self._pred_fun = self.estimator.predict

        if sample_weight is None:
            sample_weight = self._default_sample_weight(y)

        # fit and predict estimators on num_mps minipatches
        test_preds = 0 if type == 'response' else []
        importances = np.zeros((self.train_p, self.num_mps))
        importances[:] = np.nan
        for k in tqdm(range(self.num_mps)):
            self._fit_mp(X, y, sample_weight, save_estimator=False)
            idx_p = self.mp_features_[k]
            if type == 'response':
                test_preds += self._pred_fun(X_test[:, idx_p]) / self.num_mps
            elif type == 'all':
                test_preds.append(
                    copy.deepcopy(self._pred_fun(X_test[:, idx_p]))
                )
            if self.importance_fn is not None:
                importances[idx_p, k] = self.importance_fn(self.estimator)
        if self.importance_fn is not None:
            self.feature_importances_all_ = importances
            self.feature_importances_ = np.nanmean(importances, axis=1)

        # compute OOB predictions
        preds_all = copy.deepcopy(self.predictions_)
        for k, idx_n in enumerate(self.mp_samples_):
            preds_all[k][idx_n] = np.nan
        self.oob_predictions_ = np.nanmean(preds_all, axis=0)
        self.oob_score_ = self._default_score(y, self.oob_predictions_)
        
        # compute test predictions
        if isinstance(self.estimator, ClassifierMixin):
            if type == 'response':
                test_proba_preds = test_preds
                test_preds = self.classes_[np.argmax(test_preds, axis=1)]
            else:
                test_proba_preds = test_preds
                test_preds = np.array([
                    self.classes_[np.argmax(test_pred, axis=1)] for test_pred in test_preds
                ]).T
            return test_preds, test_proba_preds
        else:
            return test_preds

    def get_mp_samples(self):
        """
        Get the indicator matrix of samples used in each minipatch.

        Returns
        -------
        idx_mat : ndarray of shape (n_samples, num_mps)
            The indicator matrix of samples used in each minipatch.
        """
        idx_mat = np.zeros((self.train_n, self.num_mps))
        for k, idx_n in enumerate(self.mp_samples_):
            idx_mat[idx_n, k] = 1
        return idx_mat

    def get_mp_features(self):
        """
        Get the indicator matrix of features used in each minipatch.

        Returns
        -------
        idx_mat : ndarray of shape (n_features, num_mps)
            The indicator matrix of features used in each minipatch.
        """
        idx_mat = np.zeros((self.train_p, self.num_mps))
        for k, idx_p in enumerate(self.mp_features_):
            idx_mat[idx_p, k] = 1
        return idx_mat

    def get_feature_importance(self, rank=False):
        """
        Compute feature importance scores.

        Parameters
        ----------
        importance_fn : callable, optional
            The function to compute the feature importance scores for each minipatch.
        rank : bool, default=False
            Whether to rank the feature importance scores.

        Returns
        -------
        importances : ndarray of shape (n_features,)
            The importance scores.
        """

        if self.feature_importances_all_ is not None:
            importances = self.feature_importances_all_
        else:
            check_is_fitted(self, "estimators_")
            importances = np.zeros((self.train_p, self.num_mps))
            importances[:] = np.nan
            for k, (estimator, idx_p) in enumerate(zip(self.estimators_, self.mp_features_)):
                importances[idx_p, k] = self.importance_fn(estimator)
            self.feature_importances_all_ = importances
            self.feature_importances_ = np.nanmean(importances, axis=1)
        if rank:
            ranked_importances = np.apply_along_axis(
                lambda x: rankdata(-x, nan_policy='omit'), 0, importances
            )
            result = np.nanmean(ranked_importances, axis=1)
        else:
            result = self.feature_importances_
        result_df = pd.DataFrame({"var": range(self.train_p), "importance": result})
        return result_df

    def get_loco_importance(self, scoring_fn="auto", alpha=0.05, bonf=False, 
                            epsilon=0.0001, num_mp_B=10):
        """
        Compute leave-one-covariate-out (LOCO) minipatch importance scores.

        Parameters
        ----------
        scoring_fn : callable or str, default='auto'
            If 'auto', use the default scoring function for the estimator.
            For regressors, the deafult is the absolute residual. For classifiers, the
            default is the absolute difference between the predicted probability of the 
            true class and 1. If callable, use the scoring function provided.
        alpha : float, default=0.05
            The significance level.
        bonf : bool, default=False
            Whether to apply Bonferroni correction for multiple testing.
        epsilon : float, default=0.0001
            The normalization constant in the variance barrier calculation.
        num_mp_B : int, default=10
            The number of minipatch pairs to use for computing the
            constant B in the variance barrier.

        Returns
        -------
        loco_inf : DataFrame of shape (n_features, 4)
            The LOCO minipatch inference results, with columns
            "pval_onesided", "pval_twosided", "lower_ci", "upper_ci"
            giving the one-sided pvalue, two-sided pvalue, lower confidence
            interval, and upper confidence interval for each covariate.
        """
        predictions = self.predictions_
        mp_samples_idx = self.get_mp_samples()
        mp_features_idx = self.get_mp_features()
        if scoring_fn == "auto":
            scoring_fn = self._default_loco_scorer()

        # compute LOO/LOCO predictions
        loo_preds_diff = np.zeros(self.train_n)
        if predictions[0].ndim == 1:
            loo_preds = np.zeros(self.train_n)
            loco_preds = np.zeros((self.train_n, self.train_p))
        else:
            loo_preds = np.zeros((self.train_n, predictions[0].shape[1]))
            loco_preds = np.zeros((self.train_n, self.train_p, predictions[0].shape[1]))
        for i in range(self.train_n):
            out_samples = mp_samples_idx[i, :] == 0
            loo_mp_idxs = np.argwhere(out_samples).reshape(-1)
            loo_preds[i] = np.mean(
                [predictions[mp_idx][i] for mp_idx in loo_mp_idxs],
                axis=0
            )
            if epsilon > 0:  # for computing variance barrier
                loo_mp_idxs_subset = np.random.choice(
                    loo_mp_idxs, size=num_mp_B * 2, replace=False
                ).reshape((num_mp_B, 2))
                predictions_subset1 = np.array([
                    predictions[mp_idx][i] for mp_idx in loo_mp_idxs_subset[:, 0]
                ])
                predictions_subset2 = np.array([
                    predictions[mp_idx][i] for mp_idx in loo_mp_idxs_subset[:, 1]
                ])
                loo_preds_diff[i] = np.square(
                    predictions_subset1 - predictions_subset2
                ).mean()
            for j in range(self.train_p):
                out_features = mp_features_idx[j, :] == 0
                loco_mp_idxs = np.argwhere(out_samples & out_features).reshape(-1)
                loco_preds[i, j] = np.mean(
                    [predictions[mp_idx][i] for mp_idx in loco_mp_idxs],
                    axis=0
                )

        y_mat = np.repeat(self.y.reshape((self.train_n, 1)), self.train_p, axis=1)
        loco_resids = scoring_fn(y_mat, loco_preds)
        loo_resids = np.repeat(
            scoring_fn(self.y, loo_preds).reshape(self.train_n, 1), self.train_p, axis=1
        )
        loco_diff = loco_resids - loo_resids
        self.loo_resids_ = loo_resids
        self.loco_resids_ = loco_resids
        self.locomp_scores_ = loco_diff
        self.loo_preds_diff_ = loo_preds_diff

        # compute variance barrier
        min_var = self._get_variance_barrier(epsilon)
        self.min_var = min_var

        # do inference
        self.locomp_inf_ = np.zeros((self.train_p, 4))
        for j in range(self.train_p):
            self.locomp_inf_[j] = self._get_locomp_inf(
                loco_diff[:, j], alpha=alpha, n_tests=self.train_p, 
                bonf=bonf, min_var=min_var
            )
        self.locomp_inf_ = pd.DataFrame(
            self.locomp_inf_,
            columns=["pval_onesided", "pval_twosided", "lower_ci", "upper_ci"]
        )
        self.locomp_inf_.index.name = 'var'
        self.locomp_inf_.reset_index(inplace=True)
        return self.locomp_inf_

    def _get_locomp_inf(self, z, alpha=0.05, n_tests=1, bonf=False, min_var=0):
        """
        Compute inference for LOCO minipatch importance scores via normal approximation.

        Parameters
        ----------
        z : ndarray of shape (n_samples,)
            The LOCO minipatch importance scores.
        alpha : float, default=0.05
            The significance level.
        n_tests : int, default=1
            The number of tests being performed.
        bonf : bool, default=False
            Whether to apply Bonferroni correction for multiple testing.
        min_var : float, default=0
            The minimum variance barrier.

        Returns
        -------
        inf : ndarray of shape (4,)
            The inference results, with entries
            "pval_onesided", "pval_twosided", "lower_ci", "upper_ci"
            giving the one-sided pvalue, two-sided pvalue, lower confidence
            interval, and upper confidence interval.
        """
        try:
            s = np.nanstd(z)
        except:
            return [0, 0, 0, 0]   # should this be [1, 1, 0, 0]
        if s == 0:
            return [0, 0, 0, 0]   # should this be [1, 1, 0, 0]

        n = np.sum(~np.isnan(z))
        m = np.nanmean(z)
        sigma = s / np.sqrt(n) + min_var
        pval1 = 1 - norm.cdf(m / sigma)  # one-sided
        pval2 = 2 * (1 - norm.cdf(np.abs(m / sigma)))  # two-sided

        # Apply Bonferroni correction for M tests
        if bonf:
            pval1 = min(n_tests * pval1, 1)
            pval2 = min(n_tests * pval2, 1)
            alpha = alpha / n_tests
        q = norm.ppf(1 - alpha / 2)
        left_ci = m - q * sigma
        right_ci = m + q * sigma
        return [pval1, pval2, left_ci, right_ci]

    def _get_mp_idxs(self, sample_weight):
        """
        Get the indices of the samples and features to use in the minipatch.

        Parameters
        ----------
        sample_weight : ndarray of shape (n_samples,)
            The sample weights.

        Returns
        -------
        idx_n : ndarray of shape (n_mp_samples,)
            The indices of the samples to use in the minipatch.
        idx_p : ndarray of shape (n_mp_features,)
            The indices of the features to use in the minipatch.
        """
        if self.n_ratio == 'sqrt':
            n_mp = int(np.sqrt(self.train_n))
        else:
            n_mp = int(self.train_n * self.n_ratio)
        if self.p_ratio == 'sqrt':
            p_mp = int(np.sqrt(self.train_p))
        else:
            p_mp = int(self.train_p * self.p_ratio)
        idx_n = np.sort(np.random.choice(self.train_n, n_mp, replace=False, p=sample_weight))
        idx_p = np.sort(np.random.choice(self.train_p, p_mp, replace=False))
        return idx_n, idx_p
    
    def _get_variance_barrier(self, epsilon):
        """
        Compute the variance barrier for minipatch inference.

        Parameters
        ----------
        epsilon : float
            The normalization constant in the variance barrier calculation.

        Returns
        -------
        min_var : float
            The minimum variance barrier.
        """
        if self.n_ratio == 'sqrt':
            n_ratio = np.sqrt(self.train_n) / self.train_n
        else:
            n_ratio = self.n_ratio
        min_var = np.sqrt(np.mean(self.loo_preds_diff_)) *\
            np.log(self.train_n) * n_ratio * epsilon
        return min_var

    @abstractmethod
    def _default_score(self, y_true, y_pred):
        """
        Default scoring function to score/evaluate
        the minipatch predictions.
        """
        pass

    @abstractmethod
    def _default_sample_weight(self, y):
        """
        Compute the default sample weights.
        """
        pass

    @abstractmethod
    def _default_loco_scorer(self):
        """
        Default scoring function to score/evaluate
        the LOCO minipatch importance scores.
        """
        pass


class MinipatchRegressor(_MinipatchBase, RegressorMixin):
    """
    Minipatch regressor.
    """
    def _default_score(self, y_true, y_pred):
        """
        Compute MSE. Default scorer to evaluate minipatch predictions.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            The true responses.
        y_pred : ndarray of shape (n_samples,)
            The predicted responses.

        Returns
        -------
        mse : float
            The mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)

    def _default_sample_weight(self, y):
        """
        Uniform sample weights. Default sample weights for minipatch fitting.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The observed responses.

        Returns
        -------
        weights : ndarray of shape (n_samples,)
            The sample weights.
        """
        return np.ones(len(y)) / len(y)

    def _default_loco_scorer(self):
        """
        Function to compute absolute residuals. Default scoring function to 
        score/evaluate the LOCO minipatch importance scores.

        Returns
        -------
        scoring_fn : callable
            The scoring function.
        """
        def scoring_fn(y_true, y_pred):
            return np.abs(y_true - y_pred)
        return scoring_fn


class MinipatchClassifier(_MinipatchBase, ClassifierMixin):
    """
    Minipatch classifier.
    """

    def predict(self, X, type='response'):
        """
        Predict on the given test data.

        Parameters
        ----------
        X : ndarray of shape (n_test_samples, n_features)
            The covariate matrix for the test data.
        type : str, one of {'response', 'all'}
            If 'response', return the average of the predictions from each minipatch.
            If 'all', return the predictions from each minipatch.

        Returns
        -------
        predictions : ndarray of shape (n_test_samples,) if type='response', or
            ndarray of shape (n_test_samples, num_mps) if type='all'
            The predictions.
        """
        assert type in ['response', 'all']
        X = check_array(X)
        check_is_fitted(self, "estimators_")
        if type == 'response':
            if not hasattr(self.estimators_[0], "predict_proba"):
                raise AttributeError("'{}' object has no attribute 'predict_proba'".format(
                    self.estimators_[0].__class__.__name__)
                )
            predictions = 0
            for estimator, idx_p in zip(self.estimators_, self.mp_features_):
                predictions += self._validate_classes(
                    estimator.predict_proba(X[:, idx_p]),
                    estimator.classes_
                )
            predictions = self.classes_[np.argmax(predictions, axis=1)]
        elif type == 'all':
            predictions = []
            for estimator, idx_p in zip(self.estimators_, self.mp_features_):
                predictions.append(estimator.predict(X[:, idx_p]))
            predictions = np.array(predictions).T
        return predictions
    
    def predict_proba(self, X, type='response'):
        """
        Predict class probabilities on the given test data.

        Parameters
        ----------
        X : ndarray of shape (n_test_samples, n_features)
            The covariate matrix for the test data.
        type : str, one of {'response', 'all'}
            If 'response', return the average of the class probability predictions from each minipatch.
            If 'all', return the class probability predictions from each minipatch.

        Returns
        -------
        predictions : ndarray of shape (n_test_samples, n_classes) if type='response', or
            num_mps-length list of ndarrays of shape (n_test_samples, n_classes) if type='all'
            The class probability predictions.
        """
        assert type in ['response', 'all']
        X = check_array(X)
        check_is_fitted(self, "estimators_")
        if not hasattr(self.estimators_[0], "predict_proba"):
            raise AttributeError("'{}' object has no attribute 'predict_proba'".format(
                self.estimators_[0].__class__.__name__)
            )
        if type == 'response':
            predictions = 0
            for estimator, idx_p in zip(self.estimators_, self.mp_features_):
                predictions += self._validate_classes(
                    estimator.predict_proba(X[:, idx_p]),
                    estimator.classes_
                )
            predictions = predictions / self.num_mps
        elif type == 'all':
            predictions = []
            for estimator, idx_p in zip(self.estimators_, self.mp_features_):
                predictions.append(
                    self._validate_classes(
                        estimator.predict_proba(X[:, idx_p]),
                        estimator.classes_
                    )
                )
        return predictions
    
    def _validate_classes(self, predictions, classes):
        """
        Reorder the predictions from an estimator to match the order 
        of the minipatch estimator classes.
        
        Parameters
        ----------
        predictions : ndarray of shape (n_samples, n_classes)
            The class probability predictions.
        classes : ndarray of shape (n_classes,)
            The classes of the estimator.

        Returns
        -------
        reordered_preds : ndarray of shape (n_samples, n_classes)
            The reordered class probability predictions.
        """
        if np.array_equal(self.classes_, classes):
            return predictions
        else:
            reordered_preds = np.zeros(
                (predictions.shape[0], len(self.classes_))
            )
            for i, c in enumerate(self.classes_):
                if c in classes:
                    reordered_preds[:, i] = predictions[:, np.where(classes == c)[0]][:, 0]
            return reordered_preds

    def _default_score(self, y_true, y_pred):
        """
        Compute accuracy. Default scorer to evaluate minipatch predictions.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            The true responses.
        y_pred : ndarray of shape (n_samples,)
            The predicted responses.

        Returns
        -------
        acc : float
            The accuracy.
        """
        y_pred_classes = self.classes_[np.argmax(y_pred, axis=1)]
        return np.mean(y_true == y_pred_classes)

    def _default_sample_weight(self, y):
        """
        Compute sample weights as the inverse of the class frequencies.
        This is to achieve balanced sampling across classes.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The observed responses.

        Returns
        -------
        weights : ndarray of shape (n_samples,)
            The sample weights.
        """
        y_pd = pd.DataFrame(y.reshape((len(y), 1)))
        y_groupn = pd.DataFrame(y_pd.groupby(0).apply(lambda x: len(x)))
        y_pd = pd.merge(y_pd, y_groupn, left_on=0, right_index=True)
        weights = 1 / y_pd["0_y"].values
        return weights / np.sum(weights)

    def _default_loco_scorer(self):
        """
        Function to compute absolute difference between the predicted probability of the
        true class and 1. Default scoring function to score/evaluate the LOCO minipatch
        importance scores.

        Returns
        -------
        scoring_fn : callable
            The scoring function.
        """
        def scoring_fn(y_true, y_pred):
            score = np.zeros(y_true.shape)
            for idx, val in np.ndenumerate(y_true):
                score[idx] = np.abs(1 - y_pred[idx][y_true[idx]])
            return score
        return scoring_fn
