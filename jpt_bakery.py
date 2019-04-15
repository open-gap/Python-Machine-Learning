#%% 导入头文件
import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
print('导入头文件成功！')

#%% 读取CSV文件数据
datafram = pd.read_csv(r'D:\project/machine_learning_databases/BreadBasket_DMS.csv')
print('读取CSV文件成功！')

#%%
#datafram.head() #展示读取数据前五行
datafram.tail() #展示读取数据后五行
#datafram.info() #显示数据基本信息
datafram['Item'] = datafram['Item'].str.lower()
none_item = (datafram['Item'] == 'none').value_counts()
print('统计数据中值为None的数据个数：')
print(none_item) #输出数据为None或缺失数据的统计结果

#%% 删除异常或缺失数据
datafram.drop(datafram[datafram.Item == 'none'].index, inplace = True)
unique_len = len(datafram['Item'].unique())
# 输出Item列不同数据统计结果
print('共有', unique_len, '种不同的物品销售记录')

# #%% 绘图输出销量前10名的产品
# datafram_Items = datafram['Item'].value_counts() #分类统计卖出物品的数量
# print('The most popular items was:\n', datafram_Items.head(10)) #输出卖出物品前十的结果
# plt.figure(figsize=(13, 5)) #新建指定宽度和高度的绘图
# # 由统计的销量前十的物品名称和销量绘制条形图
# plt.bar(datafram_Items.head(10).index, datafram_Items.head(10).values)
# # 设置中文字体
# zhfont = mpl.font_manager.FontProperties(fname=r'C:\Windows/Fonts/msyh.ttc')
# plt.title('热销前十的物品统计图', fontproperties = zhfont)

# #%% 使用更为手动的方式绘制销量排行前十物品的销量统计图
# Item_array= np.arange(len(datafram_Items.head(10)))
# plt.figure(figsize=(13, 5))
# Items_name=['coffee', 'bread', 'tea', 'cake', 'pastry', 'sandwich', 'medialuna', 
# 'hot chocolate', 'cookies','brownie']
# plt.bar(Item_array, datafram_Items.head(10).iloc[:])
# plt.xticks(Item_array, Items_name) #替换横坐标数字为标签
# plt.title('Top 5 most selling items')
# plt.show()

# #%% 使用类似方式绘制饼图
# hot_items = datafram.Item.value_counts()[:10] #获取销量前十的商品信息
# other_items = datafram.Item.count() - hot_items.sum()
# item_list = hot_items.append(pd.Series([other_items], index = ['Others']))
# print('销售情况统计结果为：\n', item_list)
# values = item_list.tolist() #获取物品销量数据
# labels = item_list.index.values.tolist() #获取物品名称
# plt.figure(figsize=(10, 10))
# plt.pie(values, labels=labels, autopct='%.2f%%') #绘制饼图

#%% 利用datafram自带的绘图工具实现更简单的绘图显示效果
hot_items = datafram.Item.value_counts().head(10)
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
hot_items.plot(kind='line', title='Item most sold')
plt.subplot(2, 2, 2)
hot_items.plot(kind='bar', title='Item most sold')
plt.subplot(2, 2, 3)
hot_items.plot(kind='pie', title='Item most sold')

#%% 按不同的要求分组统计商品售卖情况
# 为了统计更为精确的时间数据，在原始数据基础上增加新的时间列
datafram['Year'] = datafram['Date'].apply(lambda x: x.split('-')[0])
datafram['Month'] = datafram['Date'].apply(lambda x: x.split('-')[1])
datafram['Day'] = datafram['Date'].apply(lambda x: x.split('-')[2])
# 统计不同时间上的数据情况，其中nunique返回不同唯一值的计数，unique返回唯一值列表
yearly_sales = datafram.groupby('Year')['Transaction'].nunique()
monthly_sales = datafram.groupby('Month')['Transaction'].nunique()
dayly_sales = datafram.groupby('Day')['Transaction'].nunique()
print('不同月份销售天数情况：')
print(datafram.groupby('Month')['Day'].nunique())
plt.figure(figsize=(13, 21)) #绘制年度、月度和每日销售数据情况
plt.subplot(3, 1, 1)
yearly_sales.plot(kind='bar', rot=0, title='Yearly Sales')
plt.subplot(3, 1, 2)
monthly_sales.plot(kind='bar', rot=0, title='Monthly Sales')
plt.subplot(3, 1, 3)
plt.bar(np.arange(31) + 1, dayly_sales, width=0.5, data=dayly_sales)
for x, y in zip(np.arange(31) + 1, dayly_sales):
    plt.text(x, y + 0.1, '%d' % y, ha='center', va='bottom', fontsize=10)
plt.title('Dayly Sales')

#%% 使用另一种方式统计销售情况的分布
datafram['Date'] = pd.to_datetime(datafram['Date']) #转化数字日期为日期类型数据
# 转化销售时间数据类型并按小时对数据进行分类
datafram['Hour'] = pd.to_datetime(datafram['Time'], format='%H:%M:%S').dt.hour
hour_sales = datafram.groupby('Hour')['Transaction'].nunique()
print('按小时统计的销售数据情况如下：')
print(hour_sales)
hour_sales = hour_sales.loc[hour_sales >= 10] #排除销售数据中少于10的偏差数据
plt.figure(figsize=(13, 5))
hour_sales.plot(kind='bar', rot=0, title='Number of Transaction made based on Hour')
datafram['Day_of_Week'] = datafram['Date'].dt.weekday #按星期对数据进行划分
weekday_sales = datafram.groupby('Day_of_Week')['Transaction'].nunique()
# 改变了星期统计数据的横坐标
weekday_sales.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
    'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(13, 5))
weekday_sales.plot(kind='bar', rot=0, 
title='Number of Transaction made based on Weekdays')

#%% 利用apriori算法分析顾客购买物品的相关性关系
# 由于缺失mlxtend包，懒得安装于是放弃，附上kaggle网址：
# https://www.kaggle.com/xvivancos/transactions-from-a-bakery