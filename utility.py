from operator import index

from pmdarima import auto_arima
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import os
import tqdm
import logging
from Exception import CustomException
import sys


# def fillMissingDates(df_df,unique_cols,frequency,freq,grain_with_time,sales_column,date_index=False):
#     date_ranges = df_df.groupby(unique_cols).apply(lambda group: pd.date_range(group[frequency].min(), group[frequency].max(), freq=freq)).reset_index(frequency)


#     # Step 4: Create a new DataFrame that combines item-location combinations with the generated date ranges
#     date_ranges = date_ranges.explode(frequency).reset_index(drop=True)


#     # Step 5: Merge the generated date ranges with the original data to get sales_quantity for each date
#     df_filled = pd.merge(date_ranges, df_df[grain_with_time+sales_column],
#                          on=unique_cols+[frequency], how='left')

#     # Step 6: Fill missing sales_quantity values with 0
#     df_filled[sales_column] = df_filled[sales_column].fillna(0)
#     if date_index:
#         return df_filled
#     # Step 7: Optionally, reset index if you want
#     else:
#         df_filled.reset_index(drop=True, inplace=True)
#         return df_filled
    
def fillMissingDates(df_df, unique_cols, frequency, freq, grain_with_time, sales_column, date_index=False):
    # Ensure `frequency` is in datetime format
    df_df[frequency] = pd.to_datetime(df_df[frequency])

    # Step 1: Generate date ranges for each unique group
    # date_ranges = (
    #     df_df.groupby(unique_cols)
    #     .apply(lambda group: pd.DataFrame({frequency: pd.date_range(group[frequency].min(), group[frequency].max(), freq=freq)}))
    #     .reset_index(level=unique_cols, drop=True)
    #     .reset_index()
    # )
    date_ranges = df_df.groupby(unique_cols).apply(
        lambda group: pd.date_range(group[frequency].min(), group[frequency].max(), freq=freq)
    ).reset_index(name=frequency)

    date_ranges = date_ranges.explode(frequency).reset_index(drop=True)
    date_ranges['date']=pd.to_datetime(date_ranges['date'])

    # Step 2: Merge generated date ranges with the original data
    df_filled = pd.merge(date_ranges, df_df[grain_with_time + sales_column], 
                         on=unique_cols + [frequency], how='left')

    # Step 3: Fill missing values in sales columns with 0
    for column in sales_column:
        df_filled[column] = df_filled[column].fillna(0)

    # Step 4: Return the DataFrame with/without resetting the index
    if date_index:
        df_filled.set_index(frequency, inplace=True)
        return df_filled
    else:
        return df_filled.reset_index(drop=True)
    
def filter_data_for_group(df,groupno,group_value=None):
    if groupno:
        new_data=df[df['default_rank']==groupno]
    elif group_value:
        new_data=df[df['comb']==group_value]
    return new_data

def MyAutoArima(data,x=None):
    arima_model=auto_arima(y=data,
    X=x,
    start_p=2,
    d=None,
    start_q=2,
    max_p=10,
    max_d=10,
    max_q=10,
    start_P=1,
    D=None,
    start_Q=1,
    max_P=10,
    max_D=10,
    max_Q=10,
    max_order=5,
    m=1,
    seasonal=True,
    stationary=False,
    information_criterion='aic',
    alpha=0.05,
    test='kpss',
    seasonal_test='ocsb',
    stepwise=False,
    n_jobs=-1,
    start_params=None,
    trend=None,
    method='lbfgs',
    maxiter=500,
    offset_test_args=None,
    seasonal_test_args=None,
    suppress_warnings=True,
    error_action='trace',
    trace=False,
    random=False,
    random_state=None,
    n_fits=100,
    return_valid_fits=False,
    out_of_sample_size=0,
    scoring='mse',
    scoring_args=None,
    with_intercept="auto")
    return arima_model


def Train_Predict(train,test,groupno):
    # filter the train and test data
    train_data=filter_data_for_group(train,groupno)
    test_data=filter_data_for_group(test,groupno)

    #filter required columns
    req_columns_train=['default_rank','comb','date','sales','onpromotion','holiday_true']
    req_columns_test=['default_rank','comb','date','onpromotion','holiday_true']
    train_data_f=train_data[req_columns_train]
    test_data_f=test_data[req_columns_test]

    # change holiday data into ints
    train_data_f.loc[:,'holiday_true']=train_data_f['holiday_true'].astype('int64')
    test_data_f.loc[:,'holiday_true']=test_data_f['holiday_true'].astype('int64')

    #fill missing dates
    train_data_ff=fillMissingDates(train_data_f,unique_cols=['default_rank'],frequency='date',freq='D',grain_with_time=['default_rank','date'],sales_column=['sales','onpromotion','holiday_true'])
    test_data_ff=fillMissingDates(test_data_f,unique_cols=['default_rank'],frequency='date',freq='D',grain_with_time=['default_rank','date'],sales_column=['onpromotion','holiday_true'])


    # min max scaler

    arima_scaler_sales = MinMaxScaler()
    arima_scaler_onpromotion = MinMaxScaler()
    
    train_data_ff['sales'] = arima_scaler_sales.fit_transform(train_data_ff[['sales']])
    # test_data_ff['sales'] = arima_scaler_sales.transform(test_data_ff[['sales']])
    
    train_data_ff['onpromotion'] = arima_scaler_onpromotion.fit_transform(train_data_ff[['onpromotion']])
    test_data_ff['onpromotion'] = arima_scaler_onpromotion.transform(test_data_ff[['onpromotion']])
    
    
    train_data_fff=train_data_ff.set_index('date')[['sales']]
    x=train_data_ff.set_index('date')[['holiday_true','onpromotion']]
    
    # handle the frq of the train and test data and exog variables data   
    train_data_fff['group']=1
    train_data_fff.reset_index(inplace=True)
    train_data_ffff=fillMissingDates(train_data_fff,unique_cols=['group'],frequency='date',freq='D',grain_with_time=['group','date'],sales_column=['sales'])
    train_data_ffff.set_index('date',inplace=True)
    train_data_ffff.index = pd.to_datetime(train_data_ffff.index)
    if train_data_ffff.index.duplicated().any():
        print("Duplicate entries found. Resolving...")
        train_data_ffff = train_data_ffff[~train_data_ffff.index.duplicated(keep='first')]
    train_data_ffff.index.freq='D'
    train_data_ffff=train_data_ffff[['sales']]
    

    x['group']=1
    x.reset_index(inplace=True)
    x=fillMissingDates(x,unique_cols=['group'],frequency='date',freq='D',grain_with_time=['group','date'],sales_column=['holiday_true','onpromotion'])
    x.set_index('date',inplace=True)
    x.index = pd.to_datetime(x.index)
    if x.index.duplicated().any():
        print("Duplicate entries found. Resolving...")
        x = x[~x.index.duplicated(keep='first')]
    x.index.freq='D'
    x=x[['holiday_true','onpromotion']]

    
    x_test=test_data_ff.set_index('date')[['holiday_true','onpromotion']]


    print("processed data")
    print(test_data.head())
    print(train_data_ffff.head())

    arima_model=MyAutoArima(train_data_ffff,x)
    res=arima_model.fit(y=train_data_ffff,X=x)
    
    start = len(train_data_ffff)
    end = start + len(x_test) - 1
    
    assert x.shape[1] == x_test.shape[1], "Mismatch in number of columns between training and test exogenous variables."
    assert len(x_test) == (end - start + 1), "Mismatch between prediction period and test exogenous variables."



    
    
    # Predict with exogenous variables
    pred_data = res.predict(n_periods=len(x_test), X=x_test).rename('sales')


    predicted_df=pd.DataFrame(pred_data)

    predicted_df['sales']=arima_scaler_sales.inverse_transform(predicted_df[['sales']]) 

    predicted_df=predicted_df.reset_index().rename(columns={'index':'date'})

    predicted_df=test_data[['id','date']].merge(predicted_df,on='date',how='left')
    predicted_df['Group']=groupno
    
    # # Output predictions
    return res,x_test,end,start,predicted_df,train_data_f,test_data
    

def Train_Predict_Prophet(train,test,groupno):
    try:
        # filter the train and test data
        train_data=filter_data_for_group(train,groupno)
        test_data=filter_data_for_group(test,groupno)

        #filter required columns
        req_columns_train=['default_rank','comb','date','sales','onpromotion','holiday_true']
        req_columns_test=['default_rank','comb','date','onpromotion','holiday_true']
        train_data_f=train_data[req_columns_train]
        test_data_f=test_data[req_columns_test]

        # change holiday data into ints
        train_data_f.loc[:,'holiday_true']=train_data_f['holiday_true'].astype('int64')
        test_data_f.loc[:,'holiday_true']=test_data_f['holiday_true'].astype('int64')

        print(test_data_f.head())
        print(train_data_f.head())

        #fill missing dates
        train_data_ff=fillMissingDates(train_data_f,unique_cols=['default_rank'],frequency='date',freq='D',grain_with_time=['default_rank','date'],sales_column=['sales','onpromotion','holiday_true'])
        test_data_ff=fillMissingDates(test_data_f,unique_cols=['default_rank'],frequency='date',freq='D',grain_with_time=['default_rank','date'],sales_column=['onpromotion','holiday_true'])


        # min max scaler

        Prophet_scaler_sales = MinMaxScaler()
        Prophet_scaler_onpromotion = MinMaxScaler()
        
        train_data_ff['sales'] = Prophet_scaler_sales.fit_transform(train_data_ff[['sales']])
        # test_data_ff['sales'] = Prophet_scaler_sales.transform(test_data_ff[['sales']])
        
        train_data_ff['onpromotion'] = Prophet_scaler_onpromotion.fit_transform(train_data_ff[['onpromotion']])
        test_data_ff['onpromotion'] = Prophet_scaler_onpromotion.transform(test_data_ff[['onpromotion']])

        # convert_columns into standard names

        train_data_ff.rename(columns={'date':'ds','sales':'y'},inplace=True)
        test_data_ff.rename(columns={'date':'ds'},inplace=True)
        print(train_data_ff.columns)
        
        # define the model
        print("processed data")
        print(test_data_ff.head())
        print(train_data_ff.head())
    except Exception as e:
        raise CustomException(e,sys)

    try:

        model=Prophet(daily_seasonality=True)
        # model.add_regressor('holiday_true')
        # model.add_regressor('onpromotion')
        model.fit(train_data_ff)
        future_dfs=model.make_future_dataframe(periods=len(test_data_ff))


        future_dfs = future_dfs.merge(
        test_data_ff[['ds', 'holiday_true', 'onpromotion']],
        on='ds',
        how='left'
        )
        future_dfs.fillna({'holiday_true': 0, 'onpromotion': 0}, inplace=True)




        # Predict with exogenous variables
        future_dfs=future_dfs.tail(len(test_data_ff)).head(len(test_data_ff))
        preds=model.predict(future_dfs)

        preds['yhat'] = Prophet_scaler_sales.inverse_transform(preds[['yhat']])
        preds['yhat_lower'] = Prophet_scaler_sales.inverse_transform(preds[['yhat_lower']])
        preds['yhat_upper'] = Prophet_scaler_sales.inverse_transform(preds[['yhat_upper']])

        #

        predicted_df=preds[['ds','yhat','yhat_upper','yhat_lower']]
        test_data['date']=pd.to_datetime(test_data['date'])
        predicted_df=test_data[['id','date']].merge(predicted_df,right_on='ds',left_on='date',how='left')
        predicted_df['Group']=groupno
        return model, future_dfs, predicted_df, train_data_f, test_data
    except Exception as e:
        raise CustomException(e,sys)

    # # Output predictions
    return model,future_dfs,predicted_df,train_data_f,test_data


 # Function to save data periodically
def save_batch_data(data_list, save_path, batch_number):
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # Create directory if it doesn't exist
    
    combined_df = pd.concat(data_list, ignore_index=True)
    file_name = f"predicted_batch_prophet_v1_{batch_number}.csv"
    file_path = os.path.join(save_path, file_name)
    combined_df.to_csv(file_path, index=False)
    print(f"Batch {batch_number} saved to {file_path}")
    return combined_df

# Main processing loop with tqdm for progress bar
def process_all_groups(df_hol, test_data_hol, all_groups, save_interval=100, save_path='predicted_arima_data_v2'):
    all_predicted_data = []
    batch_counter = 0  # To track number of processed batches

    # Loop through groups with tqdm
    for i, group in enumerate(tqdm.tqdm(all_groups, desc="Processing Groups"), start=1):
        # Process group data
        res, x_test, end, start, predicted_df, train_data_f, test_data = Train_Predict(df_hol, test_data_hol, group)
        all_predicted_data.append(predicted_df)
        
        # Check if it's time to save the data
        if i % save_interval == 0:
            batch_counter += 1
            save_batch_data(all_predicted_data, save_path, batch_counter)
            all_predicted_data.clear()  # Clear the list after saving to free memory

    # Save remaining data if any after the loop ends
    if all_predicted_data:
        batch_counter += 1
        save_batch_data(all_predicted_data, save_path, batch_counter)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# def process_all_groups_Prophet(df_hol, test_data_hol, all_groups, save_interval=100, save_path='predicted_prophet_data_v1'):
#     try:
#         all_predicted_data = []
#         batch_counter = 0
#
#         for i, group in enumerate(tqdm.tqdm(all_groups, desc="Processing Groups"), start=1):
#             try:
#                 logging.info(f"Processing group {group} (Index {i})...")
#                 model, future_dfs, predicted_df, train_data_f, test_data = Train_Predict_Prophet(df_hol, test_data_hol, group)
#                 logging.info(f"Successfully processed group {group}")
#                 all_predicted_data.append(predicted_df)
#             except Exception as e:
#                 raise CustomException(e,sys)
#                 logging.error(f"Error processing group {group}: {e}")
#                 continue
#
#             if i % save_interval == 0:
#                 batch_counter += 1
#                 try:
#                     logging.info(f"Saving batch {batch_counter}...")
#                     predicted_data=save_batch_data(all_predicted_data, save_path, batch_counter)
#                     all_predicted_data.clear()
#                     logging.info(f"Batch {batch_counter} saved successfully")
#                     return predicted_data
#                 except Exception as e:
#                     logging.error(f"Error saving batch {batch_counter}: {e}")
#
#         if all_predicted_data:
#             batch_counter += 1
#             try:
#                 logging.info(f"Saving final batch {batch_counter}...")
#                 predicted_data=save_batch_data(all_predicted_data, save_path, batch_counter)
#                 return predicted_data
#                 logging.info(f"Final batch {batch_counter} saved successfully")
#             except Exception as e:
#                 logging.error(f"Error saving final batch {batch_counter}: {e}")
#     except Exception as e:
#         raise CustomException(e,sys)

def process_all_groups_Prophet(df_hol, test_data_hol, all_groups, save_interval=100,
                               save_path='predicted_prophet_data_v1'):
    try:
        all_preds=[]
        all_predicted_data = []
        batch_counter = 0

        for i, group in enumerate(tqdm.tqdm(all_groups, desc="Processing Groups"), start=1):
            try:
                logging.info(f"Processing group {group} (Index {i})...")
                model, future_dfs, predicted_df, train_data_f, test_data = Train_Predict_Prophet(df_hol, test_data_hol,
                                                                                                 group)
                logging.info(f"Successfully processed group {group}")
                all_predicted_data.append(predicted_df)
            except Exception as e:
                raise CustomException(e, sys)
                logging.error(f"Error processing group {group}: {e}")
                continue

            if i % save_interval == 0:
                batch_counter += 1
                try:
                    logging.info(f"Saving batch {batch_counter}...")
                    predicted_data = save_batch_data(all_predicted_data, save_path, batch_counter)
                    all_preds.append(predicted_data)
                    all_predicted_data.clear()
                    logging.info(f"Batch {batch_counter} saved successfully")
                except Exception as e:
                    logging.error(f"Error saving batch {batch_counter}: {e}")

        if all_predicted_data:
            batch_counter += 1
            try:
                logging.info(f"Saving final batch {batch_counter}...")
                predicted_data = save_batch_data(all_predicted_data, save_path, batch_counter)
                all_preds.append(predicted_data)
                logging.info(f"Final batch {batch_counter} saved successfully")
            except Exception as e:
                logging.error(f"Error saving final batch {batch_counter}: {e}")

        # Now the return happens after all groups are processed
        all_predicted_df = pd.concat(all_preds, ignore_index=True)
        return all_predicted_df
    except Exception as e:
        raise CustomException(e, sys)


if __name__=="__main__":
    try:
        train_filtered=pd.read_csv('df_hol_s.csv',low_memory=False)
        test_filtered=pd.read_csv('test_data_hol_s.csv',low_memory=False)
        unique_ids = pd.concat([train_filtered['default_rank'], test_filtered['default_rank']]).unique()
        print(f"all groups to be processed {unique_ids}")
        predictions=process_all_groups_Prophet(train_filtered, test_filtered, unique_ids, save_interval=1,
                                             save_path="predicted_prophet_data_v3")

        print(predictions)
        print(predictions.shape)
        print('satish')
    except Exception as e:
        CustomException(e,sys)