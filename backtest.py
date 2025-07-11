import qlib
import pandas as pd
import torch
import os
import numpy as np

# --- 配置参数 ---
# Qlib 数据目录 (请替换为您的 Qlib 数据实际路径)
# 例如: "/home/user/.qlib/qlib_data/cn_data"
QLIB_DATA_DIR = "~/.qlib/qlib_data/cn_data" 
QLIB_REGION = "cn" # 'cn' (中国A股) 或 'us' (美国股票)

# AlphaForge 输出参数 (必须与您运行 combine_AFF.py 时的参数一致)
# 这些参数用于定位 AlphaForge 生成的预测文件
AF_SAVE_NAME = 'test' # combine_AFF.py 中的 --save_name
AF_INSTRUMENTS = 'csi500' # combine_AFF.py 中的 --instruments
AF_TRAIN_END_YEAR = 2020 # combine_AFF.py 中的 --train_end_year
AF_SEED = 0 # combine_AFF.py 中的 --seeds 列表中的一个种子
AF_N_FACTORS = 10 # combine_AFF.py 中的 --n_factors
AF_WINDOW = 'inf' # combine_AFF.py 中的 --window (可以是整数或 'inf')

# 回测时间段 (应与 AlphaForge 训练时的测试集时间段相对应)
# 例如，如果 AF_TRAIN_END_YEAR 是 2020，则测试集通常是 2022 年
BACKTEST_START_TIME = f"{AF_TRAIN_END_YEAR + 2}-01-01"
BACKTEST_END_TIME = f"{AF_TRAIN_END_YEAR + 2}-12-31"

# 交易策略参数
INITIAL_ACCOUNT = 3000000 # 初始账户资金 (例如，1亿人民币)
BENCHMARK_CODE = "SH000905" # 基准指数代码 (例如，沪深300)
TOPK_STOCKS = 50 # 每日等权重持有“超级阿尔法得分”最高的前 50 只股票
N_DROP_STOCKS = 5 # 每日最多更换 5 只股票 (控制换手率)

# 交易所相关参数 (参考 Qlib 文档和 AlphaForge 论文中的设置)
EXCHANGE_KWARGS = {
    "limit_threshold": 0.095, # 涨跌停限制 (例如，中国A股通常是 0.095 或 0.1)
    "deal_price": "close", # 交易价格 (收盘价)
    "open_cost": 0.0000954, # 开仓成本 (例如，万分之五)
    "close_cost": 0.0010954, # 平仓成本 (例如，万分之十五)
    "min_cost": 0, # 最低交易成本
}

# --- 1. Qlib 初始化 ---
print("1. 初始化 Qlib 框架...")
try:
    qlib.init(provider_uri=QLIB_DATA_DIR, region=QLIB_REGION)
    print("Qlib 初始化成功。")
except Exception as e:
    print(f"Qlib 初始化失败: {e}")
    print("请确保 QLIB_DATA_DIR 指向了正确的 Qlib 数据路径，并且数据已准备好。")
    exit()

# --- 2. 加载 AlphaForge 预测结果 ---
print("\n2. 加载 AlphaForge 生成的预测结果...")
# 根据 AlphaForge 的输出路径规则构建文件路径
# 假设 combine_AFF.py 的输出保存在其运行目录下的 'out' 文件夹中
af_pred_file_name = f"pred_{AF_TRAIN_END_YEAR}_{AF_N_FACTORS}_{AF_WINDOW}_{AF_SEED}.pt"
af_output_base_dir = f"out/{AF_SAVE_NAME}_{AF_INSTRUMENTS}_{AF_TRAIN_END_YEAR}_{AF_SEED}"
af_pred_path = os.path.join(af_output_base_dir, af_pred_file_name)

if not os.path.exists(af_pred_path):
    print(f"错误：未找到 AlphaForge 预测文件：{af_pred_path}")
    print("请确保 AlphaForge 的 'combine_AFF.py' 脚本已经运行，并且输出文件存在于指定位置。")
    exit()

# 加载 PyTorch 张量并转换为 NumPy 数组
try:
    pred_tensor = torch.load(af_pred_path).cpu().numpy()
    print(f"成功加载预测张量，形状: {pred_tensor.shape}")
except Exception as e:
    print(f"加载预测张量失败: {e}")
    exit()

# --- 3. 将预测结果转换为 Qlib 回测所需的信号格式 ---
print("\n3. 将预测结果转换为 Qlib 信号格式 (pandas.DataFrame)...")
# Qlib 回测需要一个 MultiIndex (datetime, instrument) 的 DataFrame 作为信号
# 我们需要重新加载 Qlib 数据，以获取正确的日期和股票列表进行索引映射

# 导入 AlphaForge 中用于加载 Qlib 数据的工具
# 假设 alphagen_generic.features.target 已定义，否则需要手动定义一个简单的目标表达式
try:
    from alphagen_generic.features import target
except ImportError:
    print("错误：无法导入 alphagen_generic.features.target。")
    exit()

# 使用 get_data_by_year 加载数据，以获取日期和股票 ID 映射
# 这些日期范围应覆盖 AlphaForge 训练时的所有数据，以确保正确映射
from gan.utils.data import get_data_by_year # 假设 gan 模块在 Python 路径中

try:
    returned_data = get_data_by_year(
        train_start=2012, # 确保包含所有历史数据
        train_end=AF_TRAIN_END_YEAR,
        valid_year=AF_TRAIN_END_YEAR + 1,
        test_year=AF_TRAIN_END_YEAR + 2,
        instruments=AF_INSTRUMENTS,
        target=target,
        freq='day',
    )
    # data_all_qlib 是一个 StockData 对象，包含 _dates 和 _stock_ids
    data_all_qlib, _, _, _, _, _, _ = returned_data
except Exception as e:
    print(f"加载 Qlib 数据进行索引映射失败: {e}")
    print("请检查 gan.utils.data.get_data_by_year 的导入和参数是否正确。")
    exit()

# 提取回测时间段对应的日期和股票 ID
# pred_tensor 对应的是 test_year 的数据
test_start_date_ts = pd.Timestamp(BACKTEST_START_TIME)
test_end_date_ts = pd.Timestamp(BACKTEST_END_TIME)

# 从 data_all_qlib 中筛选出回测时间段内的日期
test_dates_for_backtest = data_all_qlib._dates[
    (data_all_qlib._dates >= test_start_date_ts) & (data_all_qlib._dates <= test_end_date_ts)
]
stock_ids_for_backtest = data_all_qlib._stock_ids

# 检查预测张量和实际数据维度是否匹配
# pred_tensor 的第一维是天数，第二维是股票数
if pred_tensor.shape[0] != len(test_dates_for_backtest) or pred_tensor.shape[1] != len(stock_ids_for_backtest):
    print(f"错误：预测张量形状 {pred_tensor.shape} 与回测数据维度 ({len(test_dates_for_backtest)} 天, {len(stock_ids_for_backtest)} 股票) 不匹配。")
    print("这可能导致信号与股票数据错位。请检查 AlphaForge 的 combine_AFF.py 输出是否与 Qlib 数据加载的回测期完全对应。")
    # # 尝试进行扁平化并检查总元素数量，以防维度不完全匹配但总数一致
    # if pred_tensor.size != len(test_dates_for_backtest) * len(stock_ids_for_backtest):
    #     print("致命错误：预测张量总元素数量与期望的 (天数 * 股票数) 不匹配。无法创建信号 DataFrame。")
    #     exit()
    # else:
    #     print("尝试根据总元素数量重新整形预测张量。")
    #     pred_tensor_flat = pred_tensor.flatten()
    exit()
else:
    pred_tensor_flat = pred_tensor.flatten()

# 创建 MultiIndex
multi_index = pd.MultiIndex.from_product([test_dates_for_backtest, stock_ids_for_backtest], names=['datetime', 'instrument'])

# 创建预测信号 DataFrame
pred_score = pd.DataFrame({'score': pred_tensor_flat}, index=multi_index)

# 移除信号中的 NaN 值，因为 Qlib 策略可能无法处理
pred_score = pred_score.dropna()

print(f"预测信号 DataFrame 创建完成，形状: {pred_score.shape}")
print("信号 DataFrame 头部预览:")
print(pred_score.head())

# --- 4. 定义 Qlib 交易策略并执行回测 ---
print("\n4. 定义 Qlib 交易策略并执行回测...")
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest_daily, risk_analysis

# 定义策略配置
STRATEGY_CONFIG = {
    "topk": TOPK_STOCKS,
    "n_drop": N_DROP_STOCKS,
    "signal": pred_score, # 将我们准备好的预测信号传入策略
}

# 创建策略对象
try:
    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
    print("交易策略对象创建成功。")
except Exception as e:
    print(f"创建策略对象失败: {e}")
    exit()

# 执行回测
print(f"开始回测，时间范围从 {BACKTEST_START_TIME} 到 {BACKTEST_END_TIME}...")
try:
    report_normal, positions_normal = backtest_daily(
        start_time=BACKTEST_START_TIME,
        end_time=BACKTEST_END_TIME,
        strategy=strategy_obj,
        account=INITIAL_ACCOUNT,
        benchmark=BENCHMARK_CODE,
        exchange_kwargs=EXCHANGE_KWARGS,
    )
    print("回测执行完成。")
except Exception as e:
    print(f"回测执行失败: {e}")
    exit()

# --- 5. 评估回测结果 ---
print("\n5. 评估回测结果并生成报告...")
analysis = dict()

# 计算不含成本的超额收益
analysis["excess_return_without_cost"] = risk_analysis(
    report_normal["return"] - report_normal["bench"]
)
# 计算含成本的超额收益
analysis["excess_return_with_cost"] = risk_analysis(
    report_normal["return"] - report_normal["bench"] - report_normal["cost"]
)

# 将分析结果合并为一个 DataFrame
analysis_df = pd.concat(analysis)

print("\n--- 最终回测报告 ---")
print(analysis_df)

print("\n脚本执行完毕。您可以在上方查看回测的各项性能指标。")