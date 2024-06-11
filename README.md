# project-4
# consumer credit score 

# Final Project
    # Objective: Develop a categorical and linear regression machine learning model to analyze 
    # credit scores and predict consumer loan eligibility based on historical lending data, 
    # with the goal of assisting financial institutions in making informed decisions on loan 
    # approvals and minimizing credit risk.


    # The relationship between payment history and credit card utilization is crucial in determining an individual's creditworthiness and overall financial health. Here are some key points to consider when exploring this relationship:

    # Credit Score Impact: Payment history and credit card utilization are two significant factors that influence credit scores. Payment history accounts for a significant portion of a credit score, typically around 35%, while credit card utilization ratio (the amount of credit used compared to the total credit available) accounts for about 30%. Timely payments and low credit card utilization can positively impact credit scores.

    # Financial Responsibility: A history of making on-time payments reflects financial responsibility and reliability. Lenders and credit card companies view individuals with a positive payment history as lower credit risks. Similarly, maintaining a low credit card utilization ratio demonstrates responsible credit management and can contribute to a positive credit profile.

    # Creditworthiness: Lenders assess payment history and credit card utilization to evaluate an individual's creditworthiness when applying for loans, mortgages, or new credit cards. Consistently making timely payments and keeping credit card balances low indicate a borrower's ability to manage credit responsibly, increasing their chances of loan approval and favorable terms.

    # Debt Management: Payment history and credit card utilization are indicators of how well an individual manages their debt. Making late payments or maxing out credit cards can signal financial distress and may lead to higher interest rates, lower credit limits, and potential credit score damage. Effective debt management involves maintaining a positive payment history and using credit cards responsibly.

    # Impact on Interest Rates: Individuals with a strong payment history and low credit card utilization ratios are likely to qualify for lower interest rates on loans and credit cards. Lenders may offer preferential rates to borrowers with a history of on-time payments and prudent credit card usage, leading to cost savings over time.

    # Credit Limit Considerations: High credit card utilization can negatively impact credit scores and signal a higher risk of default. Keeping credit card balances low relative to credit limits demonstrates financial discipline and can help individuals maintain a healthy credit profile. Monitoring credit card utilization and payment history is essential for managing credit effectively.

# Part one: Importing Dependencies
    # import pandas as pd
    # from sklearn.metrics 
    # import confusion_matrix, classification_report
    # from pandas import DataFrame
    # from scipy.stats import linregress
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # import plotly.graph_objects as go
    # import warnings
    # # from xgboost import XGBClassifier
    # from sklearn.metrics import classification_report import joblib
    # #Packages options sns.set(rc={'figure.figsize': [14, 7]}, font_scale=1.2) # Standard 
    # #figure size for all np.seterr(divide='ignore', invalid='ignore', over='ignore’) ;
    # import warnings warnings.filterwarnings("ignore")
    
# Part two: Reading in data files
    # Credit Score Classification Dataset.csv
    # Credit Score Classification Dataset 164 data rows
    # Credit Risk Dataset 32k data rows
    # Credit Scoring  1k data rows


# Part three: Use get_dummies for indicator variables
    # When working with data that contains categorical variables or features, it is essential # to use a categorical machine learning model or technique.

    # Dummy variables are binary (0 or 1) variables that represent the categories of a categorical variable. Each category is converted into a separate binary column, where 1 # indicates the presence of that category and 0 indicates the absence.

    # credit_class_df['Credit Score'] = credit_class_df['Credit Score'].apply( lambda x: 2 if # x in ['High'] else (1 if x in ['Average'] else 0) )

    # credit_dummies = pd.get_dummies(credit_class_df) 
    # credit_dummies.head(15)

    #credit_dummies = pd.get_dummies(credit_class_df) 
    # credit_dummies.head(15)
    # X, y = credit_dummies.drop('Credit Score',axis=1).values , credit_dummies['Credit Score']
    
# Part four: Importing Dependencies
    # from imblearn.over_sampling import SMOTE
    # rus = SMOTE(sampling_strategy='auto')
    # X_data_rus, y_data_rus = rus.fit_resample(X, y)

    # X_train, X_test, y_train, y_test = train_test_split(X_data_rus, y_data_rus, test_size=0.3, random_state=42,stratify=y_data_rus)

    # scalar = PowerTransformer(method='yeo-johnson', standardize=True).fit(X_train)

    # X_train = scalar.transform(X_train)
    # X_test = scalar.transform(X_test)

# Part five: Importing Dependencies
    # import xgboost as xgb
    # bagging = BaggingClassifier(n_jobs=-1)
    # extraTrees = ExtraTreesClassifier(max_depth=10, n_jobs=-1)
    # randomForest = RandomForestClassifier(n_jobs=-1)
    # histGradientBoosting = HistGradientBoostingClassifier()
    # XGB = xgb.XGBClassifier(n_jobs=-1)

    # model.fit(X_train, y_train)
   
# Part five: Importing Dependencies
    # from sklearn.model_selection import train_test_split, GridSearchCV
    # from sklearn.metrics import classification_report

# The relationship between credit card utilization and density can be influenced by various factors related to consumer behavior, economic conditions, and geographical considerations. Here are some points to consider when exploring this relationship:

    # Consumer Spending Patterns: In areas with higher population density, there may be more opportunities for retail and commercial activities. This can lead to increased consumer spending and higher credit card utilization rates. The availability of shops, restaurants, entertainment venues, and other businesses in densely populated areas can contribute to higher credit card usage.

    # Convenience and Accessibility: Densely populated areas often have better access to banking services, ATMs, and online payment options. This convenience can encourage residents to use credit cards for everyday transactions, leading to higher utilization rates. The ease of making electronic payments in urban areas may contribute to increased credit card usage.

    # Cost of Living: Population density can be correlated with the cost of living in an area. In regions with higher population density, the cost of living, including housing costs, transportation expenses, and daily necessities, may be higher. Residents may rely more on credit cards to manage their expenses, resulting in higher credit card utilization rates.

    # Income Levels: Population density can also be associated with income levels and socioeconomic factors. In urban areas with higher population density, there may be a mix of income levels, including higher-income individuals who may use credit cards for rewards and benefits. This can impact credit card utilization rates in densely populated areas.

    # Debt Management: High population density areas may have diverse demographics and varying financial behaviors. Some residents may effectively manage their credit card usage and maintain low utilization rates, while others may carry high balances. The overall debt management practices in densely populated areas can influence credit card utilization patterns.

    # Financial Infrastructure: The presence of financial institutions, credit card companies, and payment processing services in densely populated areas can also influence credit card utilization rates. Residents in urban areas may have access to a wide range of financial products and services, leading to increased credit card usage.

    # Understanding the relationship between credit card utilization and population density can provide insights into consumer behavior, spending habits, and financial trends in different geographical areas. Analyzing these factors can help financial institutions, policymakers, and businesses tailor their services and strategies to meet the needs of diverse populations.
    
# An accuracy rate of 1.0 (or 100%) in a confusion matrix means that the model has predicted all instances correctly. In other words, the model has made no mistakes in its predictions when compared to the actual values in the test dataset.

    # Components of a confusion matrix in the context of accuracy:

    # True Positives (TP): The number of correct positive predictions.
    # True Negatives (TN): The number of correct negative predictions.
    # False Positives (FP): The number of incorrect positive predictions.
    # False Negatives (FN): The number of incorrect negative predictions.

    # When the accuracy rate is 1.0, it means that the sum of true positives and true negatives is equal to the total number of instances, indicating that the model has made no errors in its predictions.

