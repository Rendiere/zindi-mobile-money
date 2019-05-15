def extract_targets(train_df):
    X = train_df.drop(['mobile_money', 'savings', 'borrowing', 'insurance', 'mobile_money_classification'], axis=1)
    y = train_df['mobile_money_classification']
    return X, y
