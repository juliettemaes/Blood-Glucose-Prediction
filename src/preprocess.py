from sklearn.model_selection import train_test_split

train["time"] = pd.to_datetime(train["time"], format='%H:%M:%S')
train['time_min'] = train['time'].dt.hour * 60 + train['time'].dt.minute

test["time"] = pd.to_datetime(train["time"], format='%H:%M:%S')
test['time_min'] = test['time'].dt.hour * 60 + test['time'].dt.minute

def actual_time(df, col_var):
    df[['hours', 'minutes']] = df[col_var].str.extract('(\d+):(\d+)')
    df['time_diff'] = df['hours'].astype(int) * 60 - df['minutes'].astype(int)
    
    df['actual_time'] = df['time_min'] + df['time_diff']
        
    df['radians'] = (df["actual_time"] / 1440) * (2 * np.pi)
    
    df['sin_time'] = np.sin(df['radians'])
    df['cos_time'] = np.cos(df['radians'])
    
    df.drop(columns=['radians', 'hours', 'minutes', 'time_diff'], inplace=True)
    
    return df

variables = ['bg', 'ins', 'carbs', 'cals', 'hr', 'steps', 'activity']

X_df = pd.DataFrame()

for var in variables:
    cols = [col for col in train.columns if col.startswith(var)]
    cols_test = [col for col in test.columns if col.startswith(var)]
    var_df = pd.melt(train, id_vars=['id', 'time_min'], value_vars=cols, var_name='time', value_name=var)
    var_df_test = pd.melt(test, id_vars=['id', 'time_min'], value_vars=cols_test, var_name='time', value_name=var)

    if var == 'bg':
        def fill_na_values(group):
            # For the remaining NA values, use the mean of adjacent non-NA values
            group['bg'] = group['bg'].interpolate(method='linear')
            
            # # Backward fill the first NA values
            group['bg'] = group['bg'].bfill()
            # Forward fill the last NA values
            group['bg'] = group['bg'].ffill()
            
            return group
        
        var_df = var_df.groupby('id').apply(fill_na_values).reset_index(drop=True)
        var_df_test = var_df_test.groupby('id').apply(fill_na_values).reset_index(drop=True)
        
        target_scaler = MinMaxScaler()
        var_df[var] = target_scaler.fit_transform(var_df[[var]])
        var_df_test[var] = target_scaler.transform(var_df_test[[var]])
        
        Y_train = var_df[var_df['time'] == 'bg+1:00']
        Y_train = Y_train[["id",var]]
        Y_train = Y_train.sort_values('id').reset_index(drop=True)[var]
    
        var_df = var_df[var_df['time'] != 'bg+1:00']
        
    elif var == 'activity':
        activity_mapping3 = {'Indoor climbing': 2, 
                    'Run': 3, 
                    'Strength training': 2, 
                    'Swim': 3, 
                    'Bike': 3, 
                    'Dancing': 2, 
                    'Stairclimber': 2, 
                    'Spinning': 2, 
                    'Walking': 1, 
                    'HIIT': 3, 
                    'Outdoor Bike': 3, 
                    'Walk': 1, 
                    'Aerobic Workout': 1, 
                    'Tennis': 3, 
                    'Workout': 3, 
                    'Hike': 2, 
                    'Zumba': 2, 
                    'Sport': 2, 
                    'Yoga': 1, 
                    'Swimming': 3, 
                    'Weights': 3, 
                    'Running': 3,
                    'NaN': 0}
        
        var_df[var] = var_df[var].fillna('NaN')
        var_df[var] = var_df[var].map(activity_mapping3)
        var_df[var] = var_df[var].replace(0, np.nan)

        var_df_test[var] = var_df_test[var].fillna('NaN')
        var_df_test[var] = var_df_test[var].map(activity_mapping3)
        var_df_test[var] = var_df_test[var].replace(0, np.nan)

    else:
        scaler = MinMaxScaler()
        var_df[var] = scaler.fit_transform(var_df[[var]])
        var_df_test[var] = scaler.transform(var_df_test[[var]])

    if X_df.empty:
        X_df = var_df.copy()
        X_df_test = var_df_test.copy()
    else:
        X_df = X_df.merge(var_df, on=['id', 'time_min', 'time'], how='left')
        X_df_test = X_df_test.merge(var_df_test, on=['id', 'time_min', 'time'], how='left')

# Train complet de Kaggle
X_df = actual_time(X_df, 'time')
X_df = X_df.drop(["time_min", "time"], axis = 1)
X_df = X_df.fillna(-0.01)
X_df = X_df.sort_values('id').reset_index(drop=True)

# Test complet de Kaggle
X_df_test = actual_time(X_df_test, 'time')
X_df_test = X_df_test.drop(["time_min", "time"], axis = 1)
X_df_test = X_df_test.fillna(-0.01)
X_df_test = X_df_test.sort_values('id').reset_index(drop=True)

descriptor_layers = {}
for id_key, group in X_df.groupby('id'):
    group.drop(columns=['id'], inplace=True)
    group.set_index('actual_time', inplace=True)
    descriptor_layers[id_key] = group

descriptor_layers_test = {}
for id_key, group in X_df_test.groupby('id'):
    group.drop(columns=['id'], inplace=True)
    group.set_index('actual_time', inplace=True)
    descriptor_layers_test[id_key] = group

arrays = []
for id_key, group_df in descriptor_layers.items():
    arrays.append(group_df.values)

# Set de train complet de kaggle, format matrice 3D
X_train_kaggle = np.stack(arrays)

# Split du train de Kaggle en un train et un test set
x_train, x_test, y_train, y_test = train_test_split(X_train_kaggle, Y_train, test_size=0.2, random_state=42)


arrays = []
for id_key, group_df in descriptor_layers_test.items():
    arrays.append(group_df.values)

# Set de test de kaggle, format matrice 3D
X_test_kaggle = np.stack(arrays)