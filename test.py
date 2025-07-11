from qlib.data import D
from alphagen_qlib.stock_data import StockData

instruments_str = 'csi500'
start_time = '2022-12-31'
end_time = '2022-12-31'

# 获取 csi500 的 instruments 配置
instruments = D.instruments(market=instruments_str)

# 列出指定日期范围内的 instruments
# as_list=True 会返回一个股票代码列表
csi500_stocks = D.list_instruments(instruments=instruments, start_time=start_time, end_time=end_time, as_list=True)

data = StockData(csi500_stocks, start_time, end_time, raw = True, qlib_path = '~/.qlib/qlib_data/cn_data_rolling', freq='day')
print(f'stocks: {data._stock_ids}')
print(f'dates: {data._dates}')

# features = ['$' + f.name.lower() for f in self._features]
#         if self.raw and self.freq == 'day':
#             features = change_to_raw(features)
#         elif self.raw:
#             features = change_to_raw_min(features)

# self.fields = self._parse_fields_info(config)
# df = D.features(instruments, exprs, start_time, end_time, freq=freq, inst_processors=inst_processors)