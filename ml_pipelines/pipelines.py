#!/usr/bin/env python3
"""
Class to define machine learning pipelines for classification and clustering tasks.
"""

import logging

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class MLPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "Random Forest": RandomForestClassifier(random_state=self.random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state),
            "K-Nearest Neighbors": KNeighborsClassifier(),
        }
        self.params_grids = {
            "Decision Tree": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "Random Forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["auto", "sqrt", "log2"],
            },
            "Gradient Boosting": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 1],
                "max_depth": [3, 5, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["auto", "sqrt", "log2"],
            },
            "K-Nearest Neighbors": {
                "n_neighbors": [3, 5, 10],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
        }

    def get_cv(self, n_splits=5):
        """Returns a StratifiedKFold cross-validation object."""
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

    def baseline(self, X, y, scoring="accuracy"):
        """
        Baseline pipeline: cross-validates each model and then fits it.
        Returns a dict with:
          "results": dict of cross_val scores,
          "saved_models": fitted models,
          other keys set to None.
        """
        cv = self.get_cv()
        results = {}
        saved_models = {}

        for model_name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            score_mean = scores.mean()
            logging.info(f"{model_name} {scoring}: {score_mean:.4f}")
            results[model_name] = score_mean
            model.fit(X, y)
            saved_models[model_name] = model

        return {
            "results": results,
            "saved_models": saved_models,
            "selected_features": None,
            "best_score": score_mean,
            "best_params": None,
            "fitted_model": None,
            "feature_importances": None,
            "cluster_labels": None,
            "silhouette_score": None,
        }

    def feature_selection(self, X, y, num_features, model_name, scoring="accuracy"):
        """
        Feature selection pipeline using SequentialFeatureSelector.
        Returns a dict with:
          "selected_features": results dictionary from feature selection,
          other keys set to None.
        """
        from sklearn.feature_selection import SequentialFeatureSelector

        cv = self.get_cv()
        selected_features_dict = {}

        base_model = self.models.get(model_name)
        if base_model is None:
            raise ValueError(f"Model '{model_name}' is not defined.")

        for k in range(1, num_features + 1):
            logging.info(f"Processing {k} features")
            # Create a fresh instance for each run
            model_instance = type(base_model)(**base_model.get_params())
            sfs = SequentialFeatureSelector(
                model_instance,
                n_features_to_select=k,
                direction="forward",
                cv=cv,
                n_jobs=-1,
                scoring=scoring
            )
            sfs.fit(X, y)
            selected_features = sfs.get_support(indices=True)
            X_selected = X.iloc[:, selected_features] # TEST
            scores = cross_val_score(model_instance, X_selected, y, cv=cv, scoring=scoring, n_jobs=-1)
            score_mean = scores.mean()
            logging.info(f"{scoring} with {k} features: {score_mean:.4f}")
            result_dict = {
                "selected_features": selected_features,
                scoring: score_mean,
                "fitted_model": sfs.estimator
            }
            if hasattr(sfs.estimator, "feature_importances_"):
                result_dict["feature_importances"] = sfs.estimator.feature_importances_
            else:
                result_dict["feature_importances"] = None
            selected_features_dict[k] = result_dict

        return {
            "results": None,
            "saved_models": None,
            "selected_features": selected_features_dict,
            "best_score": None,
            "best_params": None,
            "fitted_model": None,
            "feature_importances": None,
            "cluster_labels": None,
            "silhouette_score": None,
        }

    def hp_search(self, X, y, model_name, scoring="accuracy"):
        """
        Hyperparameter search pipeline using GridSearchCV.
        Returns a dict with:
          "best_score", "best_params", "fitted_model" (best estimator),
          "feature_importances" if available,
          other keys set to None.
        """
        from sklearn.model_selection import GridSearchCV

        base_model = self.models.get(model_name)
        if base_model is None:
            raise ValueError(f"Model '{model_name}' is not defined.")

        cv = self.get_cv()
        param_grid = self.params_grids.get(model_name)
        search = GridSearchCV(
            base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        search.fit(X, y)
        best_score = search.best_score_
        best_params = search.best_params_
        best_estimator = search.best_estimator_

        logging.info(f"{model_name} - Best parameters: {best_params}")
        logging.info(f"{model_name} - Best {scoring}: {best_score:.4f}")

        feature_importances = best_estimator.feature_importances_ if hasattr(best_estimator, "feature_importances_") else None

        return {
            "results": None,
            "saved_models": None,
            "selected_features": None,
            "best_score": best_score,
            "best_params": best_params,
            "fitted_model": best_estimator,
            "feature_importances": feature_importances,
            "cluster_labels": None,
            "silhouette_score": None,
        }

    def feature_selection_hp_search(self, X, y, num_features, model_name, scoring="accuracy"):
        """
        Combined feature selection and hyperparameter search pipeline.
        Returns a dict with:
          "selected_features": combined results (dict keyed by number of features),
          other keys set to None.
        """
        feature_selection_results = self.feature_selection(X, y, num_features, model_name, scoring)["selected_features"]
        combined_results = {}

        for k, fs_result in feature_selection_results.items():
            logging.info(f"Performing hyperparameter search on {k} selected features")
            selected_features = fs_result["selected_features"]
            X_selected = X[:, selected_features]
            hp_search_result = self.hp_search(X_selected, y, model_name, scoring)
            logging.info(f"{model_name} - Best parameters with {k} features after HP search: {hp_search_result['best_params']}")
            logging.info(f"{model_name} - Best {scoring} with {k} features after HP search: {hp_search_result['best_score']:.4f}")

            # Merge feature selection and hp search results for current k
            combined_results[k] = {
                "selected_features": selected_features,
                scoring: hp_search_result["best_score"],
                "best_params": hp_search_result["best_params"],
                "fitted_model": hp_search_result["fitted_model"],
                "feature_importances": hp_search_result["feature_importances"]
            }

        return {
            "results": None,
            "saved_models": None,
            "selected_features": combined_results,
            "best_score": None,
            "best_params": None,
            "fitted_model": None,
            "feature_importances": None,
            "cluster_labels": None,
            "silhouette_score": None,
        }

    def unsupervised(self, X, n_clusters=2):
        """
        Unsupervised pipeline using KMeans and silhouette score.
        Returns a dict with:
          "silhouette_score", "cluster_labels",
          other keys set to None.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        logging.info(f"KMeans with {n_clusters} clusters - Silhouette score: {silhouette_avg:.4f}")

        return {
            "results": None,
            "saved_models": None,
            "selected_features": None,
            "best_score": None,
            "best_params": None,
            "fitted_model": None,
            "feature_importances": None,
            "cluster_labels": cluster_labels,
            "silhouette_score": silhouette_avg,
        }