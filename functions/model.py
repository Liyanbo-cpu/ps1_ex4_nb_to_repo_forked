import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def map_categorical_to_numeric(df, column):
    """
    Map categorical string values to numeric values (e.g., 'female': 1, 'male': 0).
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        column (str): The column to map.
        
    Returns:
        pd.DataFrame: Dataframe with the mapped column.
    """
    if column == 'Sex':
        df[column] = df[column].map({'female': 1, 'male': 0}).astype(int)
    return df

def split_data(df, predictors, target, test_size=0.2, random_state=42):
    """
    Split the data into train and validation sets.
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
        predictors (list): List of feature column names.
        target (str): The target column name.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed.
        
    Returns:
        tuple: (train_X, train_Y, valid_X, valid_Y)
    """
    train, valid = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    
    train_X = train[predictors]
    train_Y = train[target].values
    valid_X = valid[predictors]
    valid_Y = valid[target].values
    
    return train_X, train_Y, valid_X, valid_Y

def train_and_predict(train_X, train_Y, valid_X):
    """
    Initialize, train, and predict using a RandomForestClassifier.
    
    Parameters:
        train_X: Training features.
        train_Y: Training labels.
        valid_X: Validation features.
        
    Returns:
        tuple: (preds_train, preds_valid)
    """
    clf = RandomForestClassifier(n_jobs=-1, 
                                 random_state=42,
                                 criterion="gini",
                                 n_estimators=100,
                                 verbose=False)
    
    clf.fit(train_X, train_Y)
    
    preds_train = clf.predict(train_X)
    preds_valid = clf.predict(valid_X)
    
    return preds_train, preds_valid

def evaluate_model(train_Y, preds_train, valid_Y, preds_valid):
    """
    Print classification reports for both training and validation data.
    """
    print("Classification report for training data:")
    print(metrics.classification_report(train_Y, preds_train, target_names=['Not Survived', 'Survived']))
    
    print("\nClassification report for validation data:")
    print(metrics.classification_report(valid_Y, preds_valid, target_names=['Not Survived', 'Survived']))