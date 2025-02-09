import pandas as pd

def  read_csv_as_numpy(input_data):
    '''
    CSVファイルを読み込み, NumPy配列に変換する関数
    '''
    df = pd.read_csv(input_data, header=None)
    np_array = df.to_numpy()

    return np_array