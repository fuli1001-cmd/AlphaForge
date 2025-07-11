import qlib
from qlib.data import D
from qlib.data.ops import Feature  # 导入 Feature 类
import datetime
import pandas as pd

# 初始化 Qlib
# 假设数据已下载到 ~/.qlib/qlib_data/cn_data
provider_uri = "~/.qlib/qlib_data/cn_data_rolling"
# 确保 Qlib 初始化时设置了正确的区域，并处理可能的线程安全问题（Windows 下需要）
# 如果您在 Linux 或 macOS 上运行，可以省略 if __name__ == "__main__":
# 但为了兼容性，建议保留
if __name__ == "__main__":
    qlib.init(provider_uri=provider_uri, region=qlib.constant.REG_CN)

    # 定义日期范围
    start_time = "2021-08-05"
    end_time = "2023-02-20"

    print(f"Retrieving data for CSI500 stocks from {start_time} to {end_time}")

    # 获取 csi500 的 instruments 配置
    instruments = D.instruments(market='csi500')

    # 列出指定日期范围内的 instruments
    # as_list=True 会返回一个股票代码列表
    csi500_stocks = D.list_instruments(instruments=instruments, start_time=start_time, end_time=end_time, as_list=True)

    print(f"Number of CSI500 stocks found: {len(csi500_stocks)}")
    print(f"First 10 CSI500 stocks: {csi500_stocks[:10]}")

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
    df = df.swaplevel().sort_index()
    df = df.stack().unstack(level=1)
    dates = df.index.levels[0]
    stock_ids = df.columns
    print(f'stocks: {stock_ids}')
    print(f'dates: {dates}')

    # calendar = D.calendar(start_time=start_time, end_time=end_time, freq="day")
    # dates_with_full_data = []

    # for date in calendar:
    #     print(f"---- {date} ----")
    #     # 获取这些股票的特征数据
    #     df = D.features(csi500_stocks, fields, start_time=date, end_time=date, freq='day')
    #     print(len(df))

    #     # 打印df中$volume为空或0的行
    #     # print(df[df['$volume'].isna() | (df['$volume'] == 0)])

    #     # if len(df) != len(csi500_stocks):
    #     #     print(f"---- {date} ----")
    #     #     # dates_with_full_data.append(date)
    #     #     # 打印不在df的index(instrument)中的stock
    #     #     missing_stocks = set(csi500_stocks) - set(df.index.get_level_values('instrument'))
    #     #     print(f"Missing stocks on {date}: {missing_stocks}")

    # print(f"Found {len(dates_with_full_data)} days in 2022 where all {len(csi500_stocks)} stocks have data.")
    # if dates_with_full_data:
    #     print(f"First 5 dates: {[d.strftime('%Y-%m-%d') for d in dates_with_full_data[:5]]}")

    # # print("\nSample of the retrieved data:")
    # # print(data.head())
    # # print(df.tail())
    # # print("\nData description:")
    # # print(data.describe())

