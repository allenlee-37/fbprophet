import os
import pandas as pd
from matplotlib import pyplot
from prophet import Prophet
from datetime import datetime
import argparse
import unicodedata

# 시작 날짜 & 예측 끝 날짜
start_date='2023-01-01'
end_date='2028-12-31'

# test
region = '무안군'

def get_csv_list(region):
    dir_region = f'./raw_data/{region}'
    file_list = os.listdir(dir_region)
    result = []
    for elem in file_list:
        if '.csv' in elem: result.append(elem)
    return result

def concat_csv(region, csv_list):
    df1 = pd.read_csv(f'./raw_data/{region}/{csv_list[0]}', encoding='cp949')
    for i in range(1, len(csv_list)):
        df2 = pd.read_csv(f'./raw_data/{region}/{csv_list[0]}', encoding='cp949')
        # df2 = df2.map(lambda x: unicodedata.normalize('NFC', x))
        df1 = pd.concat([df1, df2], ignore_index=True)
        df = df1[df1['방문자 구분']=='외부방문자(b+c)']
        df = df[['기준년월', '방문자 수']]
        df.columns = ['기준연월', '방문자 수']
    return df.reset_index(drop=True)

def get_date(df):
    dataset = pd.DataFrame({'date': df['기준연월'],
                   'year': list(map(lambda x: int(x/100), df['기준연월'])),
                   'month': list(map(lambda x: int(x%100), df['기준연월'])),
                   'visitor': df['방문자 수']})
    print('- 날짜 입력')
    dataset['ds'] = pd.to_datetime({'year':dataset['year'], 'month':dataset['month'], 'day':1})
    print('- 컬럼 명 변경')
    dataset.rename(columns={'visitor':'y'}, inplace=True)
    return dataset

def limitations(df):
    cap = df['y'].max()*100
    if df['y'].min() > 100: 
        floor = df['y'].min() / 100
    else: 
        floor = df['y'].min()
    return floor, cap

def holidays():
    df = pd.DataFrame({
        'holiday': 'corona',
        'ds': pd.date_range('2020-03-01', '2022-3-01'),
        'lower_window': 0,
        'upper_window': 1,
        })
    return df

def train_model(df, floor, cap, exceptions):
    df = df[['ds', 'y']]
    df['floor'] = floor
    df['cap'] = cap
    model = Prophet(changepoint_prior_scale=0.1,
                interval_width = 0.7, 
                holidays=exceptions, 
                #holidays_prior_scale = 0.1,
                growth='logistic', 
                seasonality_mode='multiplicative', 
                #changepoints = ['2013-01-01', '2018-01-01', '2020-03-01', '2022-03-01']
                )
    model.fit(df)
    return model, df


def predict_model(model, df, floor, cap, start_date, end_date):
    future = pd.concat([df[['ds']], pd.DataFrame(pd.date_range(start_date, end_date, freq = 'MS'), columns = ['ds'])])
    future['floor'] = floor
    future['cap'] = cap
    return model.predict(future)

def save_prediction(prediction, city):
    result_dir = f'/Users/master/dev/PythonPr/fbProphet/result/{city}'
    try:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    except OSError:
        print('Error: Failed to create directory.')

    prediction.to_excel(f'{result_dir}/full_result.xlsx')
    y_only = prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    y_only.to_excel(f'{result_dir}/y_only.xlsx')
    y_only['year'] = [i.year for i in y_only['ds']]
    y_only['month'] = [i.month for i in y_only['ds']]
    prediction_yearly = pd.pivot_table(y_only, index = 'year', values = ['yhat', 'yhat_lower', 'yhat_upper'], aggfunc = 'sum')
    prediction_yearly.to_excel(f'{result_dir}/yearly.xlsx')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--region', '-r', type=str, required=True)
    args = p.parse_args()

    region = args.region
    csv_list = get_csv_list(region)
    
    df = concat_csv(region, csv_list)
    df = get_date(df)
    floor, cap = limitations(df)
    exceptions = holidays()
    print('- start training')
    model, df = train_model(df, floor, cap, exceptions)
    print('- start prediction')
    prediction = predict_model(model, df, floor, cap, start_date, end_date)
    print('- start saving excel')
    save_prediction(prediction, args.region)
    print('- saved excel in result folder')
    return 

if __name__ == '__main__': main()
    
