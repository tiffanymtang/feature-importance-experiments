import copy
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from scipy.stats import norm


class _MinipatchBase(BaseEstimator):
    """
    Base class for minipatch estimators.
    """

    def __init__(self, estimator, n_ratio, p_ratio, B, random_state=None):
        self.estimator = estimator
        self.n_ratio = n_ratio
        self.p_ratio = p_ratio
        self.B = B
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model to the given training data.
        :param X: ndarray of shape (n_samples, n_features)
            The covariate matrix.
        :param y: ndarray of shape (n_samples,)
            The observed responses.
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

        n = self.train_n
        p = self.train_p
        np.random.seed(self.random_state)

        if sample_weight is None:
            sample_weight = self._get_default_sample_weight(y)

        # fit estimators on B minipatches
        for b in range(self.B):
            idx_n, idx_p = self._get_mp_idxs(sample_weight)
            X_train = X[idx_n, :][:, idx_p]
            y_train = y[idx_n]
            self.estimator.fit(X_train, y_train)
            self.estimators_.append(copy.deepcopy(self.estimator))
            self.mp_samples_.append(idx_n)
            self.mp_features_.append(idx_p)
            if isinstance(self.estimator, ClassifierMixin):
                preds = self.estimator.predict_proba(X[:, idx_p])
            else:
                preds = self.estimator.predict(X[:, idx_p])
            self.predictions_.append(preds)

        # compute OOB predictions
        preds_all = copy.deepcopy(self.predictions_)
        for b, idx_n in enumerate(self.mp_samples_):
            preds_all[b][idx_n] = np.nan
        self.oob_predictions_ = np.nanmean(preds_all, axis=0)
        self.oob_score_ = self._default_score(y, self.oob_predictions_)

    def predict(self, X, type='response'):
        assert type in ['response', 'all']
        if type == 'response':
            predictions = 0
            for estimator, idx_p in zip(self.estimators_, self.mp_features_):
                predictions += estimator.predict(X[:, idx_p])
            predictions = predictions / self.B
        elif type == 'all':
            predictions = []
            for estimator, idx_p in zip(self.estimators_, self.mp_features_):
                predictions.append(estimator.predict(X[:, idx_p]))
            predictions = np.array(predictions).T
        return predictions

    def predict_proba(self, X, type='response'):
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
                predictions += estimator.predict_proba(X[:, idx_p])
            predictions = predictions / self.B
        elif type == 'all':
            predictions = []
            for estimator, idx_p in zip(self.estimators_, self.mp_features_):
                predictions.append(estimator.predict_proba(X[:, idx_p]))
            # predictions = np.array(predictions).T
        return predictions

    def get_mp_samples(self):
        idx_mat = np.zeros((self.train_n, self.B))
        for b, idx_n in enumerate(self.mp_samples_):
            idx_mat[idx_n, b] = 1
        return idx_mat

    def get_mp_features(self):
        idx_mat = np.zeros((self.train_p, self.B))
        for b, idx_p in enumerate(self.mp_features_):
            idx_mat[idx_p, b] = 1
        return idx_mat

    def get_loco_importance(self, alpha=0.05, bonf=True):
        predictions = self.predictions_
        mp_samples_idx = self.get_mp_samples()
        mp_features_idx = self.get_mp_features()

        # compute LOCO predictions
        loco_preds = np.zeros((self.train_n, self.train_p))
        for i in range(self.train_n):
            for j in range(self.train_p):
                keep_mp_idxs = np.argwhere(
                    (mp_samples_idx[i, :] == 0) & (mp_features_idx[j, :] == 0)
                ).reshape(-1)
                loco_preds[i, j] = np.array(
                    [predictions[mp_idx][i] for mp_idx in keep_mp_idxs]
                ).mean()

        # compute LOO/LOCO predictions
        loo_preds = np.zeros(self.train_n)
        loco_preds = np.zeros((self.train_n, self.train_p))
        for i in range(self.train_n):
            out_samples = mp_samples_idx[i, :] == 0
            keep_mp_idxs = np.argwhere(out_samples).reshape(-1)
            loo_preds[i] = np.array(
                [predictions[mp_idx][i] for mp_idx in keep_mp_idxs]
            ).mean()
            for j in range(self.train_p):
                out_features = mp_features_idx[j, :] == 0
                keep_mp_idxs = np.argwhere(out_samples & out_features).reshape(-1)
                loco_preds[i, j] = np.array(
                    [predictions[mp_idx][i] for mp_idx in keep_mp_idxs]
                ).mean()

        y_mat = np.repeat(self.y.reshape((self.train_n, 1)), self.train_p, axis=1)
        loco_resids = np.abs(y_mat - loco_preds)
        loo_resids = np.repeat(
            np.abs(self.y - loo_preds).reshape(self.train_n, 1), self.train_p, axis=1
        )
        loco_diff = loco_resids - loo_resids
        self.loo_resids_ = loo_resids
        self.loco_resids_ = loco_resids
        self.locomp_scores_ = loco_diff

        # do inference
        self.locomp_inf_ = np.zeros((self.train_p, 4))
        for j in range(self.train_p):
            self.locomp_inf_[j] = self._get_locomp_inf(
                loco_diff[:, j], alpha=alpha, n_tests=self.train_p, bonf=bonf
            )
        self.locomp_inf_ = pd.DataFrame(
            self.locomp_inf_,
            columns=["pval_onesided", "pval_twosided", "lower_ci", "upper_ci"]
        )
        self.locomp_inf_.index.name = 'var'
        self.locomp_inf_.reset_index(inplace=True)
        return self.locomp_inf_

    def _get_locomp_inf(self, z, alpha=0.05, n_tests=1, bonf=True):
        try:
            s = np.nanstd(z)
        except:
            return [0, 0, 0, 0]
        if s == 0:
            return [0, 0, 0, 0]

        n = np.sum(~np.isnan(z))
        m = np.nanmean(z)
        pval1 = 1 - norm.cdf(m / s * np.sqrt(n))  # one-sided
        pval2 = 2 * (1 - norm.cdf(np.abs(m / s * np.sqrt(n))))  # two-sided

        # Apply Bonferroni correction for M tests
        if bonf:
            pval1 = min(n_tests * pval1, 1)
            pval2 = min(n_tests * pval2, 1)
            alpha = alpha / n_tests
        q = norm.ppf(1 - alpha / 2)
        left_ci = m - q * s / np.sqrt(n)
        right_ci = m + q * s / np.sqrt(n)
        return [pval1, pval2, left_ci, right_ci]

    def _get_mp_idxs(self, sample_weight):
        n_mp = int(self.train_n * self.n_ratio)
        p_mp = int(self.train_p * self.p_ratio)
        idx_n = np.sort(np.random.choice(self.train_n, n_mp, replace=False, p=sample_weight))
        idx_p = np.sort(np.random.choice(self.train_p, p_mp, replace=False))
        return idx_n, idx_p

    @abstractmethod
    def _default_score(self, y_true, y_pred):
        pass

    @abstractmethod
    def _get_default_sample_weight(self, y):
        pass



class MinipatchRegressor(_MinipatchBase, RegressorMixin):
    """
    Minipatch regressor.
    """
    def _default_score(self, y_true, y_pred):
        """
        Compute MSE.
        :param y_true:
        :param y_pred:
        :return:
        """
        return np.mean((y_true - y_pred) ** 2)

    def _get_default_sample_weight(self, y):
        return np.ones(len(y)) / len(y)


class MinipatchClassifier(_MinipatchBase, ClassifierMixin):
    """
    Minipatch classifier.
    """
    def _default_score(self, y_true, y_pred):
        """
        Compute accuracy.
        :param y_true:
        :param y_pred:
        :return:
        """
        return np.mean(y_true == y_pred)

    def _get_default_sample_weight(self, y):
        y_pd = pd.DataFrame(y.reshape((len(y), 1)))
        y_groupn = pd.DataFrame(y_pd.groupby(0).apply(lambda x: len(x)))
        y_pd = pd.merge(y_pd, y_groupn, left_on=0, right_index=True)
        weights = 1 / y_pd["0_y"].values
        return weights / np.sum(weights)