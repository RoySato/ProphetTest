import quandl
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet



# How many days of stocks it would predict after the learning process
# 学習後、何日分の予想を立てさせるか
prediction_range = 365

# API key of Quandl
# QuandlのAPIキー

quandl.ApiConfig.api_key="*****"

# TSE refers to Tokyo Stock Exchange, Inc. The number fllowing after / would identify the comapany. Fx.) 2802 refers to Rakuten Inc.
# TSEで東証。/に続いて証券コードを入れると、その企業の株価推移を見れる。
quandl_data = quandl.get("TSE/2802")
quandl_data.to_csv('Rakuten.csv')


# Keys: Date,Open,High,Low,Close,Volume
# Specify the range of the data by using negative number. -10 gets the 10 newest data.
# 何日前からいつまでのデータを取得するか
whole_data = pd.read_csv('Rakuten.csv')
whole_length = -len(whole_data['Date'])
# 日付
s_date = whole_data['Date'][whole_length: -prediction_range]
# 始値
s_open = whole_data['Open'][whole_length: -prediction_range]
# 高値
s_high = whole_data['High'][whole_length: -prediction_range]
# 安値
s_low = whole_data['Low'][whole_length: -prediction_range]
# 終値
s_close = whole_data['Close'][whole_length: -prediction_range]
# 出来高
s_volume = whole_data['Volume'][whole_length: -prediction_range]


# To use Prophet, name the axes with sd and y. In this case, dates is x and close is y
# x軸とy軸の定義をすることで、Prophetにとって扱いやすいデータに成形する
m_stock  = pd.read_csv('Rakuten.csv', skiprows=1, header=None, names=['ds','Open', 'High', 'Low','y','Volume'])

# Learning process
# 学習
model = Prophet()
model.fit(m_stock)

# periods: How many days of prediction it would make. 何日分の予測を立てるか
# freq: 'd' stands for daily. 'd'で1日ごとのデータ予測
prediction = model.make_future_dataframe(periods=prediction_range, freq = 'd')

# eliminating the weekends
# 週末分の排除
prediction = prediction[prediction['ds'].dt.weekday < 5]

# Prediction and vidualization process
# 予測と可視化
p_data = model.predict(prediction)
model.plot(p_data)

# Plotting the actual change
# 実際の動きも可視化してみる
fig2 = plt.figure()
fig = plt.figure()

axes = fig.add_axes([0.1, 0.1, 1.0, 1.0])
axes2 = fig2.add_axes([0.1, 0.1, 1.0, 1.0])
axes.plot(s_open, 'blue', lw=1)
axes.plot(s_high, 'red', lw=1)
axes.plot(s_low, 'black', lw=1)
axes.plot(s_close, 'orange', lw=1)
axes2.plot(s_volume, 'black', lw=1)

axes.set_xlabel('Dates')
axes.set_ylabel('Price')
axes.set_title('Rakuten')
axes.legend()
axes2.set_xlabel('Dates')
axes2.set_ylabel('Volume')
axes2.set_title('Rakuten')
axes2.legend()

plt.show()
