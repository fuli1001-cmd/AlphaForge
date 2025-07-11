import qlib
from qlib.data import D
from qlib.data.ops import Feature

provider_uri = "~/.qlib/qlib_data/cn_data_rolling"

if __name__ == "__main__":
    qlib.init(provider_uri=provider_uri, region=qlib.constant.REG_CN)

    # 定义日期范围
    start_time = "2022-01-01"
    end_time = "2022-12-31"

    print(f"Retrieving data for CSI500 stocks from {start_time} to {end_time}")

    # 获取 csi500 的 instruments 配置
    instruments = D.instruments(market='csi500')

    # 使用 Feature 对象定义要获取的字段
    # 这比直接使用字符串 '$close' 等更健壮，可以避免解析问题
    fields = [
        Feature("close"),
        Feature("volume"),
        Feature("open"),
        Feature("high"),
        Feature("low")
    ]

    df = D.features(instruments, fields, start_time=start_time, end_time=end_time, freq='day')

    # 检查原始 df 的最后几行数据
    print("\nOriginal DF Tail:")
    print(df.tail())

    # 直接从原始 MultiIndex 中获取最大日期
    latest_date = df.index.get_level_values('datetime').max()
    print(f"\nLatest date found in raw data: {latest_date}")

    df.to_csv("csi500_features_1.csv", index=True)
    df = df.swaplevel().sort_index()
    df.to_csv("csi500_features_2.csv", index=True)
    df = df.stack().unstack(level=1)
    dates = df.index.levels[0]
    stock_ids = df.columns
    print(f'get by instruments name:\n df shape: {df.shape},\n date count: {len(dates)}, stock count: {len(stock_ids)}')
    print(dates)

    # # 列出指定日期范围内的 instruments
    # # as_list=True 会返回一个股票代码列表
    # csi500_stocks = D.list_instruments(instruments=instruments, start_time=start_time, end_time=end_time, as_list=True)
    # print(f"Number of CSI500 stocks found: {len(csi500_stocks)}")
    # df = D.features(csi500_stocks, fields, start_time=start_time, end_time=end_time, freq='day')
    # df = df.swaplevel().sort_index()
    # df = df.stack().unstack(level=1)
    # dates = df.index.levels[0]
    # stock_ids = df.columns
    # print(f'get by instruments name:\n df shape: {df.shape},\n date count: {len(dates)}, stock count: {len(stock_ids)}')

