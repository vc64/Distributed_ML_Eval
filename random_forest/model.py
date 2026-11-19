def train_forest(features, labels, n_estimators=50, random_state=42):
    from sklearn.ensemble import RandomForestClassifier

    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")

    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=1,
    )
    return forest.fit(features, labels)
