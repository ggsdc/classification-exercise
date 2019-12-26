
def prepare_data(data_df):
    data_df['male'] = data_df.apply(lambda row: row['gender'] == "Male", axis=1)
    data_df['female'] = data_df.apply(lambda row: row['gender'] == "Female", axis=1)
    data_df['partner'] = data_df.apply(lambda row: row['Partner'] == "Yes", axis=1)
    data_df['dependents'] = data_df.apply(lambda row: row['Dependents'] == "Yes", axis=1)
    data_df['phone'] = data_df.apply(lambda row: row['PhoneService'] == "Yes", axis=1)
    data_df['mul_lines'] = data_df.apply(lambda row: row['MultipleLines'] == "Yes", axis=1)
    data_df['dsl'] = data_df.apply(lambda row: row['InternetService'] == "DSL", axis=1)
    data_df['fiber'] = data_df.apply(lambda row: row['InternetService'] == "Fiber optic", axis=1)
    data_df['security'] = data_df.apply(lambda row: row['OnlineSecurity'] == "Yes", axis=1)
    data_df['backup'] = data_df.apply(lambda row: row['OnlineBackup'] == "Yes", axis=1)
    data_df['protection'] = data_df.apply(lambda row: row['DeviceProtection'] == "Yes", axis=1)
    data_df['support'] = data_df.apply(lambda row: row['TechSupport'] == "Yes", axis=1)
    data_df['tv'] = data_df.apply(lambda row: row['StreamingTV'] == "Yes", axis=1)
    data_df['movies'] = data_df.apply(lambda row: row['StreamingMovies'] == "Yes", axis=1)
    data_df['month'] = data_df.apply(lambda row: row['Contract'] == "Month-to-month", axis=1)
    data_df['one-year'] = data_df.apply(lambda row: row['Contract'] == "One year", axis=1)
    data_df['two-year'] = data_df.apply(lambda row: row['Contract'] == "Two year", axis=1)
    data_df['paper-billing'] = data_df.apply(lambda row: row['PaperlessBilling'] == "No", axis=1)
    data_df['bank-transfer'] = data_df.apply(lambda row: row['PaymentMethod'] == "Bank transfer (automatic)", axis=1)
    data_df['credit-card'] = data_df.apply(lambda row: row['PaymentMethod'] == "Credit card (automatic)", axis=1)
    data_df['electronic-check'] = data_df.apply(lambda row: row['PaymentMethod'] == "Electronic check", axis=1)
    data_df['mailed-check'] = data_df.apply(lambda row: row['PaymentMethod'] == "Mailed check", axis=1)
    data_df['total-charges'] = data_df.apply(lambda row: float(row['TotalCharges']), axis=1)
    data_df['churn'] = data_df.apply(lambda row: row['Churn'] == "Yes", axis=1)

    col_drop = ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService', 'MultipleLines', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod','TotalCharges', 'Churn']

    data_df = data_df.drop(columns=col_drop)

    return data_df
