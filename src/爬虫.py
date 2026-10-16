import baostock as bs
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

"""
登陆Baostock系统
注意事项：
1. 无需输入账号密码，系统默认公共账号
2. 调用任何接口前都需要先登录
3. 使用完毕需要登出以释放资源
"""
lg = bs.login()
# 检查登录状态（0表示成功）
print('login respond error_code:' + lg.error_code)
print('login respond error_msg:' + lg.error_msg)

"""
获取股票历史K线数据
参数说明：
- code: 股票代码，sh.600000 代表上证浦东发展银行
- fields: 指定返回的字段（日期、开盘价、收盘价等）
- start_date/end_date: 日期区间（注意此处原代码日期顺序错误，已修正）
- frequency: 数据频率，d=日k线，w=周，m=月
- adjustflag: 复权类型，3=不复权，1=前复权，2=后复权
"""
# 自动校正日期顺序（如果开始日期晚于结束日期）
start_date, end_date = sorted(['2020-01-20', '2020-07-01'])  # 用户数据日期范围需要调整
rs = bs.query_history_k_data_plus(
    code="sh.600000",
    fields="date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
    start_date=start_date,
    end_date=end_date,
    frequency="d",
    adjustflag="3"
)

# 检查数据请求状态
print('\nquery_history_k_data_plus respond error_code:' + rs.error_code)
print('query_history_k_data_plus respond error_msg:' + rs.error_msg)

"""
数据处理与存储
1. 将数据转换为Pandas DataFrame
2. 类型转换（字符串转数值）
3. 按时间排序（确保时间序列正确）
"""
data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())

if data_list:
    result = pd.DataFrame(data_list, columns=rs.fields)
    # 转换数据类型（将字符串转为数值）
    numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
    result[numeric_columns] = result[numeric_columns].apply(pd.to_numeric, errors='coerce')
    # 转换日期格式并排序
    result['date'] = pd.to_datetime(result['date'])
    result.sort_values('date', inplace=True)

    # 保存到CSV（注意路径中的中文可能存在编码问题）
    save_path = r"history_A_stock_k_data.csv"
    result.to_csv(save_path, index=False)
    print(f"\n数据已保存至：{save_path}")

    """
    数据可视化模块
    绘制价格趋势图（收盘价）和成交量双轴图表
    """
    plt.figure(figsize=(14, 7))

    # 创建双轴
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # 绘制收盘价曲线
    ax1.plot(result['date'], result['close'],
             label='Close Price', color='tab:blue', linewidth=1.5)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (RMB)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 绘制成交量柱状图
    ax2.bar(result['date'], result['volume'],
            label='Volume', color='tab:orange', alpha=0.3, width=0.8)
    ax2.set_ylabel('Volume', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # 设置日期格式
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    # 添加图例和标题
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Shanghai Pudong Development Bank (600000) Stock Trend')
    plt.tight_layout()

    # 保存和显示图表
    plt.savefig('stock_trend_visualization.png', dpi=300)
    plt.show()

else:
    print("\n警告：未获取到有效数据，请检查参数设置")

"""
登出系统
说明：完成操作后必须登出来释放系统资源
"""
bs.logout()