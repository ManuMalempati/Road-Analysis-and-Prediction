import pandas as pd
from sklearn.metrics.cluster import normalized_mutual_info_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

#  Calculates NMI for any categorical column vs severity
def nmi_severity(dataset, col, nmi_dict, sub_dataset_id):
    ''' Takes the dataset, factor to be analysed, dictionary to store results and 
    subset_id (ACCIDENT_NO or PERSON_ID or VEHICLE_ID since we have to group them due
    to one to many relationships e.g accident has many people) as input and stores 
    the NMI values in the dictionary'''
    df = pd.read_csv(dataset, usecols=['ACCIDENT_NO', sub_dataset_id, col, 'SEVERITY'])
    df['SEVERITY'] = pd.to_numeric(df['SEVERITY'])

    # Groupby only if the column is originally not from the accident dataset itself
    if sub_dataset_id == 'ACCIDENT_NO':
        avg_sev = df.dropna(subset=[col, 'SEVERITY'])[['ACCIDENT_NO', col, 'SEVERITY']]
    else:
        avg_sev = df.groupby(['ACCIDENT_NO', sub_dataset_id, col], as_index=False)['SEVERITY'].mean()

    avg_sev[col] = avg_sev[col].astype('category')
    # Convert from float to integer
    avg_sev['SEVERITY'] = avg_sev['SEVERITY'].astype(int)

    nmi = normalized_mutual_info_score(avg_sev[col], avg_sev['SEVERITY'])
    nmi_dict[col] = float(nmi)
    #print(f"NMI between {col} and SEVERITY:", nmi)

# Calculates NMI for any categorical column vs injury level
def nmi_avg_injury(dataset, col, nmi_dict, sub_dataset_id):
    ''' Takes the dataset, factor to be analysed, dictionary to store results and 
    subset_id (ACCIDENT_NO or PERSON_ID or VEHICLE_ID since we have to group them due
    to one to many relationships e.g accident has many people) as input and stores 
    the NMI values in the dictionary'''

    df_inj = pd.read_csv(dataset, usecols=['ACCIDENT_NO',sub_dataset_id,col,'INJ_LEVEL'])

    # convert to numeric type
    df_inj['INJ_LEVEL'] = pd.to_numeric(df_inj['INJ_LEVEL'])

    # Groupby only if the column is originally not from the accident dataset itself
    if sub_dataset_id == 'ACCIDENT_NO':
        avg_inj = df_inj.dropna(subset=[col, 'INJ_LEVEL'])[['ACCIDENT_NO', col, 'INJ_LEVEL']]
    else:
        avg_inj = df_inj.groupby(['ACCIDENT_NO', sub_dataset_id, col], as_index=False)['INJ_LEVEL'].mean()

    # treat as categorical
    avg_inj[col] = avg_inj[col].astype('category')
    # Convert from float to integer
    avg_inj['INJ_LEVEL'] = avg_inj['INJ_LEVEL'].astype(int)

    nmi = normalized_mutual_info_score(avg_inj[col], avg_inj['INJ_LEVEL'])
    nmi_dict[col] = float(nmi)
    #print(f"NMI between {col} and INJ_LEVEL:", nmi)

# Creates chart for the given dictionary
def save_bar_chart(data, title, filename, x_label='Value'):
    """
    Creates and saves a horizontal bar chart from a given dictionary, title,
    filename to save.
    """

    # Convert the dictionary to a DataFrame and sort by value
    df = pd.DataFrame(data.items(), columns=['Label', 'Value']).sort_values(by='Value')
    # Adjust figure size based on the number of items
    plt.figure(figsize=(8, max(4, 0.3 * len(df))))

    # Create the bar plot
    sns.barplot(x='Value', y='Label', hue='Label', data=df, palette='Blues_d', legend=False)

    # Set title and axis labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('')

    # Add the values on the bars for better readability
    for index, value in enumerate(df['Value']):
        plt.text(value + 0.001, index, f"{value:.4f}", va='center', fontsize=8)

    # Extend x-axis
    plt.xlim(0, df['Value'].max() * 1.1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Chart '{filename}' saved successfully.")

# Function for analysing sub-factors influence severity
def sub_factor_analysis(dataset, col, target_col):
    '''
    Takes the dataser, factor to be analysed and target variable as input and returns a 
    dictionary with the sub-factors of the factor and their mean target values.
    '''
    # Check if the column has a description and use that for better readability
    try:
        df = pd.read_csv(dataset, usecols=[col+'_DESC', target_col])
        df = df.dropna()
        # Calculate the mean target value for each unique sub-catgeory
        grouped = df.groupby(col+'_DESC')[target_col].mean()
    except ValueError:
        df = pd.read_csv(dataset, usecols=[col, target_col])
        df = df.dropna()
        # Calculate the mean target value for each unique sub-catgeory
        grouped = df.groupby(col)[target_col].mean()

    # Convert to dict and sort by the values
    grouped = grouped.to_dict()
    grouped = dict(sorted(grouped.items(), key=lambda x: x[1], reverse=True))
    return grouped

# Main Function to perform correlation analysis
def correlation_analysis(dataset):
    '''Performs correlation analysis on the dataset and stores results in json files and
    charts in png format'''

    os.makedirs('./correlation_results', exist_ok=True)

    print("Performing correlation analysis....")

    # dictionaries for NMI scores for severity
    nmi_dict_severity = {}

    # dictionaries for NMI scores for injuiry level for q1
    nmi_dict_inj_1 = {}

    # dictionaries for NMI scores for injuiry level for q2
    nmi_dict_inj_2 = {}

    # factors to be considered for research question 1
    factors = [
        'ACCIDENT_TYPE', 'DAY_OF_WEEK', 'DCA_CODE', 'LIGHT_CONDITION',
        'ROAD_GEOMETRY', 'SPEED_ZONE', 'RMA', 'TIME_OF_DAY',

        'AGE_GROUP', 'SEX', 'ROAD_USER_TYPE', 'HELMET_BELT_WORN',
        'SEATING_POSITION',
        
        'VEHICLE_TYPE', 'VEHICLE_AGE',
        'TARE_WEIGHT', 'SEATING_CAPACITY',
        'FUEL_TYPE', 'CONSTRUCTION_TYPE',
        'VEHICLE_DCA_CODE', 'CAUGHT_FIRE',
        'TRAFFIC_CONTROL',

        'ATMOSPH_COND'
    ]

    ''' Some factors like POLICE_ATTEND, 'NO_PERSONS_INJ_3', 'NO_PERSONS_NOT_INJ' and 
    'NO_PERSONS_KILLED', 'NO_PERSONS_INJ_2' and 'CAUGHT_FIRE' are not included in the 
    correlation analysis since they are post-accident factors and not pre-accident factors.'''

    for factor in factors:
        nmi_severity(dataset, factor, nmi_dict_severity, 'ACCIDENT_NO')
        nmi_avg_injury(dataset, factor, nmi_dict_inj_1, 'ACCIDENT_NO')

    # factors from research question 2 (directly from person.csv)
    factors_q2 = ['HELMET_BELT_WORN', 'SEATING_POSITION', 'VEHICLE_TYPE']

    # Calculate NMI for factors from q2 vs injury level
    for factor in factors_q2:
        nmi_avg_injury(dataset, factor, nmi_dict_inj_2, 'PERSON_ID')

    # Sort the dictionaries by NMI values
    sorted_sev_1 = dict(sorted(nmi_dict_severity.items(), key=lambda x:x[1], reverse=True))
    sorted_inj_1 = dict(sorted(nmi_dict_inj_1.items(), key=lambda x: x[1], reverse=True))
    sorted_inj_2 = dict(sorted(nmi_dict_inj_2.items(), key=lambda x:x[1], reverse=True))

    # Store the NMI values in JSON output
    import json
    with open('./correlation_results/nmi_results.json', 'w') as f:
        json.dump({'factors vs severity': sorted_sev_1, 
                'factors vs injury_level_q1': sorted_inj_1, 
                'factors vs injury_level_q2': sorted_inj_2}
                , f, indent=2)

    print('nmi_results.json file created successfully.')

    # Now we analyse top 4 factors from q1 that influence severity
    sub_factor_dicts = {}
    sub_factor_dicts['factors vs severity'] = {}
    sub_factor_dicts['factors vs injury_level_q1'] = {}
    sub_factor_dicts['factors vs injury_level_q2'] = {}

    for factor in list(sorted_sev_1.keys())[:4]:
        sub_factor_dicts['factors vs severity'][factor] = sub_factor_analysis(dataset, 
                                                                            factor, 
                                                                            'SEVERITY')
    # Now we analyse top 4 factors from q1 that influence injury level
    for factor in list(sorted_inj_1.keys())[:4]:
        sub_factor_dicts['factors vs injury_level_q1'][factor] = sub_factor_analysis(dataset, 
                                                                                    factor, 
                                                                                    'INJ_LEVEL'
                                                                                    )
    # Now we analyse factors (there are only 3) from q2 that influence injury level
    for factor in factors_q2:
        sub_factor_dicts['factors vs injury_level_q2'][factor] = sub_factor_analysis(dataset, 
                                                                                    factor, 
                                                                                    'INJ_LEVEL'
                                                                                    )

    # Store the sub-factor analysis results in JSON output
    with open('./correlation_results/sub_factor_analysis.json', 'w') as f:
        json.dump(sub_factor_dicts, f, indent=2)

    print('sub_factor_analysis.json file created successfully.')

    # Save all 3 charts
    save_bar_chart(sorted_sev_1, 'Q1 NMIs: Factors vs SEVERITY', './correlation_results/nmi_severity.png', 
                x_label='NMI')
    save_bar_chart(sorted_inj_1, 'Q1 NMIs: Factors vs INJURY LEVEL', './correlation_results/nmi_injury_level_q1.png',
                    x_label='NMI')
    save_bar_chart(sorted_inj_2, 'Q2 NMIs: Factors vs INJURY LEVEL', './correlation_results/nmi_injury_level_q2.png',
                    x_label='NMI')
    
    print("Correlation analysis completed successfully.")
