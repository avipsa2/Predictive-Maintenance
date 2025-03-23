import streamlit as st
import pandas as pd
import pickle
import numpy as np
model = pickle.load(open(r"C:\Users\Asus\Predictive Maintenance\new_model.pkl", "rb"))


st.title("Predictive Maintenance System")

# Upload dataset files
uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True)

if uploaded_files:
    data_dict = {}
    for file in uploaded_files:
        data_dict[file.name] = pd.read_csv(file)
    
    # Extract required datasets
    errors = data_dict.get("errors.csv")
    telemetry = data_dict.get("telemetry.csv")
    maintenance = data_dict.get("maint.csv")
    failures = data_dict.get("failures.csv")
    machines = data_dict.get("machines.csv")
    
    if all(df is not None for df in [errors, telemetry, maintenance, failures, machines]):
        r=[errors,failures,maintenance,telemetry]
        for i in r:
            i['datetime']=pd.to_datetime(i['datetime'],format="%Y-%m-%d %H:%M:%S")
            i.sort_values(["datetime", "machineID"], inplace=True, ignore_index=True)
        dummies = pd.get_dummies(errors['errorID'], drop_first=False)
        errors = errors.drop(columns=['errorID']).join(dummies)
        errors = telemetry[['datetime', 'machineID']].merge(errors, on=['machineID', 'datetime'], how='left').fillna(0.0)
        
        telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
        telemetry = telemetry.sort_values(['machineID', 'datetime'])
        
        # Compute rolling statistics
        fields = ['volt', 'rotate', 'pressure', 'vibration']
        temp = []
        fields = ['volt', 'rotate', 'pressure', 'vibration']
        for col in fields:
            temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).resample('3H', closed='left', label='right').mean().unstack())
        telemetry_mean_3h = pd.concat(temp, axis=1)
        telemetry_mean_3h.columns = [i + 'mean_3h' for i in fields]
        telemetry_mean_3h.reset_index(inplace=True)

        temp = []
        for col in fields:
            temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).resample('3H', closed='left', label='right').std().unstack())
        telemetry_sd_3h = pd.concat(temp, axis=1)
        telemetry_sd_3h.columns = [i + 'sd_3h' for i in fields]
        telemetry_sd_3h.reset_index(inplace=True)

        temp=[]
        fields = ['volt', 'rotate', 'pressure', 'vibration']
        for col in fields:
            temp.append(pd.pivot_table(telemetry,index='datetime',columns='machineID',values=col).resample('3H',closed='left',label='right').first().unstack().rolling(window=24, center=False).mean())
        telemetry_mean_24h = pd.concat(temp, axis=1)
        telemetry_mean_24h.columns = [i + 'mean_24h' for i in fields]
        telemetry_mean_24h.reset_index(inplace=True)
        telemetry_mean_24h = telemetry_mean_24h.loc[-telemetry_mean_24h['voltmean_24h'].isnull()]
        

        temp = []
        fields = ['volt', 'rotate', 'pressure', 'vibration']
        for col in fields:
            temp.append(pd.pivot_table(telemetry, index='datetime',columns='machineID',values=col).resample('3H',closed='left',label='right').first().unstack().rolling(window=24, center=False).std())
        telemetry_sd_24h = pd.concat(temp, axis=1)
        telemetry_sd_24h.columns = [i + 'sd_24h' for i in fields]
        telemetry_sd_24h.reset_index(inplace=True)
        telemetry_sd_24h = telemetry_sd_24h.loc[-telemetry_sd_24h['voltsd_24h'].isnull()]

        telemetryf = pd.concat([telemetry_mean_3h,
                            telemetry_sd_3h.iloc[:, 2:6],
                            telemetry_mean_24h.iloc[:, 2:6],
                            telemetry_sd_24h.iloc[:, 2:6]], axis=1).dropna()
        
        dummies = pd.get_dummies(maintenance['comp'], drop_first=False)
        dummies_bool = dummies.astype(int)
        maintenance=pd.concat([maintenance, dummies_bool], axis=1)
        maintenance.drop('comp',axis=1,inplace=True)
        maintenance=telemetry[['datetime', 'machineID']].merge(maintenance, on=['datetime','machineID'],how='outer').fillna(0).sort_values(by=['machineID', 'datetime'])
        components = ['comp1', 'comp2', 'comp3', 'comp4']
        for comp in components:
            maintenance.loc[maintenance[comp] < 1, comp] = None
            maintenance.loc[-maintenance[comp].isnull(),
                 comp] = maintenance.loc[-maintenance[comp].isnull(), 'datetime']
            maintenance[comp] = maintenance[comp].fillna(method='ffill')
        maintenance = maintenance.loc[maintenance['datetime'] > pd.to_datetime('2015-01-01')]
        for comp in components:
            maintenance[comp] = (maintenance["datetime"] - pd.to_datetime(maintenance[comp])) / np.timedelta64(1, "D") 

        temp = []
        fields = ['error%d' % i for i in range(1, 6)]
        for col in fields:
            pivoted = pd.pivot_table(errors, index='datetime',columns='machineID',values=col)
            resampled = pivoted.resample('3H', closed='left', label='right').first()
            resampled = resampled.dropna()
            unstacked = resampled.unstack()
            rolling_sum = unstacked.rolling(window=24, center=False).sum()
            temp.append(rolling_sum)
        error_count = pd.concat(temp, axis=1)
        error_count.columns = [i + 'count' for i in fields]
        error_count.reset_index(inplace=True)
        error_count = error_count.dropna()
        


        final = telemetryf.merge(error_count, on=['datetime', 'machineID'], how='left')
        final = final.merge(maintenance, on=['datetime', 'machineID'], how='left')
        final = final.merge(machines, on=['machineID'], how='left')
        final = final.fillna(method='ffill', limit=7).fillna(0)

        f = pd.get_dummies(final.drop(columns=['datetime', 'machineID'], axis=1))

        X = f

        predictions = model.predict(X)
        final['Failure Prediction'] = predictions
        failures_predicted=final[final['Failure Prediction']!='none']

        st.subheader("🚨 Failure Predictions")
        st.write(failures_predicted[['datetime', 'machineID', 'Failure Prediction']])
        st.download_button("Download Predictions", final.to_csv(index=False), "predictions.csv", "text/csv")
        
    else:
        st.error("Please upload all required files: errors.csv, telemetry.csv, maint.csv, failures.csv, machines.csv")
