import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

breast_msk = pd.read_csv('breast_msk_2018_clinical_data.tsv',sep='\t')
breast_msk.drop(columns = ['Study ID', 'Patient ID', 'Sample ID', 'Cancer Type',
'Tumor Tissue Origin', 'Tumor Sample Histology', 'Somatic Status', 
'Site of Sample', 'Oncotree Code'], inplace = True)
    
def label_encode_columns(df, columns):
    """
    Performs label encoding on specified columns of a DataFrame.

    Parameters:2
    df (DataFrame): Input DataFrame.
    columns (list): List of column names to be label encoded.

    Returns:
    DataFrame: DataFrame with label encoded columns.
    """
    df_encoded = breast_msk.copy()
    label_encoders = {}
        
    for col in columns:
        label_encoder = LabelEncoder()
        df_encoded[col + '_encoded'] = label_encoder.fit_transform(df[col])
        label_encoders[col] = label_encoder
        
    return df_encoded, label_encoders


columns_to_encode = [ 
        'ER Status of Sequenced Sample', 
        'ER Status of the Primary',  
        'HER2 FISH Status of Sequenced Sample', 
        'HER2 FISH Ratio Value of Sequenced Sample',
        'HER2 FISH Ratio Primary',
        'HER2 FISH Status (Report and ASCO) of Primary',
        'HER2 IHC Status Primary', 
        'HER2 IHC Score of Sequenced Sample', 
        'HER2 IHC Status of Sequenced Sample', 
        'HER2 IHC Score Primary', 
        'HER2 Primary Status', 
        'Overall HR Status of Sequenced Sample',  
        'Primary Tumor Laterality', 
        'Menopausal Status At Diagnosis', 
        'Metastatic Disease at Last Follow-up', 
        'Metastatic Recurrence Time', 
        'M Stage', 
        'N Stage',  
        'Overall Survival Status', 
        'Overall HER2 Status of Sequenced Sample', 
        'Overall Patient HER2 Status', 
        'Overall Patient HR Status', 
        'Overall Patient Receptor Status', 
        'Overall Primary Tumor Grade', 
        'Primary Nuclear Grade', 
        'Prior Breast Primary', 
        'Prior Local Recurrence', 
        'PR Status of Sequenced Sample', 
        'PR Status of the Primary', 
        'Receptor Status Primary', 
        'Number of Samples Per Patient', 
        'Sample Type', 
        'Sex', 
        'Stage At Diagnosis', 
        'Time To Death (Months)',
        'TMB (nonsynonymous)',   
        'T Stage',
        'Patient\'s Vital Status',
]

print(breast_msk['Cancer Type Detailed'].value_counts())
print(breast_msk.shape)

# removing invalid cancer type
breast_msk = breast_msk[breast_msk['Cancer Type Detailed'] != 'Breast']

# Perform label encoding
breast_msk, label_encoders = label_encode_columns(breast_msk, columns_to_encode)
    
## Dropping unwanted rows
columns_encoded = [
    'ER PCT Primary',
    'ER Status of Sequenced Sample', 
    'ER Status of the Primary',  
    'HER2 FISH Status of Sequenced Sample', 
    'HER2 FISH Ratio Value of Sequenced Sample',
    'HER2 FISH Ratio Primary',
    'HER2 FISH Status (Report and ASCO) of Primary',
    'HER2 IHC Status Primary', 
    'HER2 IHC Score of Sequenced Sample', 
    'HER2 IHC Status of Sequenced Sample', 
    'HER2 IHC Score Primary', 
    'HER2 Primary Status', 
    'Overall HR Status of Sequenced Sample',  
    'Primary Tumor Laterality', 
    'Menopausal Status At Diagnosis', 
    'Metastatic Disease at Last Follow-up', 
    'Metastatic Recurrence Time', 
    'M Stage', 
    'N Stage',  
    'Overall Survival Status', 
    'Overall HER2 Status of Sequenced Sample', 
    'Overall Patient HER2 Status', 
    'Overall Patient HR Status', 
    'Overall Patient Receptor Status', 
    'Overall Primary Tumor Grade', 
    'Primary Nuclear Grade', 
    'Prior Breast Primary', 
    'Prior Local Recurrence',
    'PR PCT Primary',
    'PR Status of Sequenced Sample', 
    'PR Status of the Primary', 
    'Receptor Status Primary', 
    'Number of Samples Per Patient', 
    'Sample Type', 
    'Sex', 
    'Stage At Diagnosis', 
    'Time To Death (Months)',
    'TMB (nonsynonymous)',   
    'T Stage',
    'Patient\'s Vital Status',
]
breast_msk.drop(columns = columns_encoded, inplace = True)

# plot Cancer Type Detailed
plt.figure(figsize=(9, 6))
plt.title('Cancer Type Detailed', weight='bold')
sns.countplot(x=breast_msk['Cancer Type Detailed'], palette='flare')
plt.xticks(rotation=45, ha='right')
plt.show()

mapper={
    'Breast Invasive Ductal Carcinoma': 0, 
    'Breast Invasive Lobular Carcinoma': 1, 
    'Breast Mixed Ductal and Lobular Carcinoma': 2,
    'Breast Invasive Cancer, NOS ': 3,
    'Metaplastic Breast Cancer': 4,
    'Breast Invasive Mixed Mucinous Carcinoma': 5,
    'Invasive Breast Carcinoma': 6,
    'Adenoid Cystic Breast Cancer': 7,
}
breast_msk['Cancer Type Detailed'] = breast_msk['Cancer Type Detailed'].map(mapper)
breast_msk = breast_msk.dropna()

# Separate features and labels
features = breast_msk.drop(columns=['Cancer Type Detailed'])
labels = breast_msk['Cancer Type Detailed']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the model using Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))