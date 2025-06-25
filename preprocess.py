import pandas as pd
import os

def preprocess_data():
    # Create an output folder for processed data
    os.makedirs('./processed', exist_ok=True)

    # Load the datasets
    accident_df = pd.read_csv('./datasets/accident.csv')
    person_df   = pd.read_csv('./datasets/person.csv')
    vehicle_df  = pd.read_csv('./datasets/vehicle.csv')
    weather_df  = pd.read_csv(
        './datasets/atmospheric_cond.csv',
        usecols=['ACCIDENT_NO', 'ATMOSPH_COND']
    )

    # Function to determine time of day from a given time
    def get_time_of_day(time):
        if pd.isnull(time):
            return None
        try:
            hour, minute = map(int, str(time)[:5].split(':'))
            total_minutes = hour * 60 + minute
            if 360 <= total_minutes <= 719:
                return 'Morning'
            elif 720 <= total_minutes <= 1079:
                return 'Afternoon'
            elif 1080 <= total_minutes <= 1439:
                return 'Evening'
            else:
                return 'Late Night'
        except:
            return None

    # Add TIME_OF_DAY
    accident_df['ACCIDENT_TIME'] = accident_df['ACCIDENT_TIME'].str.slice(0, 5)
    accident_df['TIME_OF_DAY']    = accident_df['ACCIDENT_TIME'].apply(get_time_of_day)

    # Extract YEAR
    accident_df['ACCIDENT_DATE'] = pd.to_datetime(
        accident_df['ACCIDENT_DATE'], errors='coerce'
    )
    accident_df['YEAR'] = accident_df['ACCIDENT_DATE'].dt.year

    # Merge person + accident
    merged_df = pd.merge(person_df, accident_df, on='ACCIDENT_NO', how='left')

    # Pick important vehicle columns and merge
    vehicle_cols = [
        'ACCIDENT_NO', 'VEHICLE_ID',
        'VEHICLE_TYPE', 'VEHICLE_TYPE_DESC',
        'VEHICLE_YEAR_MANUF', 'VEHICLE_WEIGHT', 'TARE_WEIGHT',
        'SEATING_CAPACITY', 'FUEL_TYPE',
        'LEVEL_OF_DAMAGE', 'CONSTRUCTION_TYPE',
        'VEHICLE_DCA_CODE',
        'CAUGHT_FIRE',
        'TRAFFIC_CONTROL', 'TRAFFIC_CONTROL_DESC'
    ]
    merged_df = pd.merge(
        merged_df,
        vehicle_df[vehicle_cols],
        on=['ACCIDENT_NO', 'VEHICLE_ID'],
        how='left'
    )

    # VEHICLE_AGE
    merged_df['VEHICLE_YEAR_MANUF'] = pd.to_numeric(
        merged_df['VEHICLE_YEAR_MANUF'], errors='coerce'
    )
    merged_df['VEHICLE_AGE'] = merged_df['YEAR'] - merged_df['VEHICLE_YEAR_MANUF']

    # Add weather
    merged_df = pd.merge(
        merged_df,
        weather_df,
        on='ACCIDENT_NO',
        how='left'
    )

    # --- MISSING-VALUE IMPUTATION MOVED HERE ---
    print("Cleaning data for analysis...")
    total_missing = merged_df.isna().sum().sum()
    print(f"Found {total_missing} missing values across {merged_df.shape[1]} columns")

    # Categorical fills
    merged_df['SEX'].fillna(
        merged_df['SEX'].mode()[0], inplace=True
    )
    print(f"Filled SEX missing values with mode: {merged_df['SEX'].mode()[0]}")

    merged_df['SEATING_POSITION'].fillna(
        merged_df['SEATING_POSITION'].mode()[0], inplace=True
    )
    print(f"Filled SEATING_POSITION missing values with mode: {merged_df['SEATING_POSITION'].mode()[0]}")

    merged_df['RMA'].fillna(
        merged_df['RMA'].mode()[0], inplace=True
    )
    print(f"Filled RMA missing values with mode: {merged_df['RMA'].mode()[0]}")

    # Numeric fills
    for col in ['HELMET_BELT_WORN', 'TARE_WEIGHT', 'VEHICLE_DCA_CODE', 'ATMOSPH_COND']:
        median_val = merged_df[col].median()
        merged_df[col].fillna(median_val, inplace=True)
        print(f"Filled {col} missing values with median: {median_val}")

    # Now all further NA handling (e.g. drop severity/injury) can proceed as before
    # Fix categorical types
    cat_columns = [
        'ACCIDENT_TYPE', 'DAY_OF_WEEK', 'DCA_CODE',
        'LIGHT_CONDITION', 'ROAD_GEOMETRY', 'RMA', 'TIME_OF_DAY',
        'AGE_GROUP', 'SEX', 'ROAD_USER_TYPE',
        'HELMET_BELT_WORN', 'SEATING_POSITION',
        'VEHICLE_TYPE', 'FUEL_TYPE', 'CONSTRUCTION_TYPE',
        'VEHICLE_DCA_CODE', 'CAUGHT_FIRE', 'TRAFFIC_CONTROL',
        'ATMOSPH_COND', 'EJECTED_CODE', 'TAKEN_HOSPITAL',
        'INJ_LEVEL'
    ]
    for col in cat_columns:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].astype(str).fillna("Unknown")

    # Numeric conversions
    num_columns = [
        'NO_OF_VEHICLES', 'NO_PERSONS_NOT_INJ', 'SPEED_ZONE',
        'INJ_LEVEL', 'SEVERITY',
        'VEHICLE_WEIGHT', 'TARE_WEIGHT',
        'SEATING_CAPACITY', 'VEHICLE_AGE'
    ]
    for col in num_columns:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    # Drop rows with missing severity or injury level
    merged_df.dropna(subset=['SEVERITY', 'INJ_LEVEL'], inplace=True)

    # Outlier removal (unchanged)
    def remove_outliers(df, column):
        if column not in df:
            return df
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    outlier_cols = [
        'NO_OF_VEHICLES','NO_PERSONS_NOT_INJ',
        'SPEED_ZONE','INJ_LEVEL','VEHICLE_AGE'
    ]
    for col in outlier_cols:
        merged_df = remove_outliers(merged_df, col)

    # Save fully imputed & cleaned data
    merged_df.to_csv('./processed/full_preprocessed.csv', index=False)
    print("Processed data saved in './processed/full_preprocessed.csv'.")
    print("Preprocessing complete.")