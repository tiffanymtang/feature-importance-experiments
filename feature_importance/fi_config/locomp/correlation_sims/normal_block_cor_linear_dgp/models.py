import copy
from sklearn.ensemble import RandomForestRegressor
from feature_importance.util import ModelConfig, FIModelConfig
from feature_importance.scripts.competing_methods import tree_mdi_plus, tree_mdi, tree_mdi_OOB, tree_mda, tree_shap, locomp
from feature_importance.scripts.minipatch import MinipatchRegressor
from imodels.importance import RandomForestPlusRegressor, RidgeRegressorPPM

rf_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.33, random_state=27)
ridge_model = RidgeRegressorPPM()
rfplus_model = RandomForestPlusRegressor(rf_model=copy.deepcopy(rf_model), prediction_model=copy.deepcopy(ridge_model))
# B = 1000
B = 10000

ESTIMATORS = [
    [ModelConfig('RF', RandomForestRegressor, model_type='rf',
                 other_params={'n_estimators': 100, 'min_samples_leaf': 5, 'max_features': 0.33, 'random_state': 27})],
    [ModelConfig('RF+', RandomForestPlusRegressor, model_type='rf+',
                 other_params={'rf_model': copy.deepcopy(rf_model),
                               'prediction_model': copy.deepcopy(ridge_model)})],
    [ModelConfig('MP+RF', MinipatchRegressor, model_type='mp',
                 other_params={'estimator': copy.deepcopy(rf_model),
                               'n_ratio': 'sqrt', 'p_ratio': 'sqrt', 'B': B})],
    [ModelConfig('MP+RF+', MinipatchRegressor, model_type='mp',
                 other_params={'estimator': copy.deepcopy(rfplus_model),
                               'n_ratio': 'sqrt', 'p_ratio': 'sqrt', 'B': B})]
]

FI_ESTIMATORS = [
    [FIModelConfig('MDI', tree_mdi, model_type='rf', splitting_strategy='train-train-test')],
    [FIModelConfig('TreeSHAP', tree_shap, model_type='rf', splitting_strategy='train-train-test')],
    [FIModelConfig('MDI+', tree_mdi_plus, model_type='rf+', splitting_strategy='train-train-test',
                   other_params={'refit': False})],
    [FIModelConfig('LOCO-MP', locomp, model_type='mp', splitting_strategy='train-train-test')],
    # [FIModelConfig('MDI-oob', tree_mdi_OOB, model_type='rf')],
    # [FIModelConfig('MDA', tree_mda, model_type='rf')],
]