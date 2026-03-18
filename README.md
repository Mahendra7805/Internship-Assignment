# Internship-Assignment
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sentiment = pd.read_csv("fear_greed_index.csv")
trader = pd.read_csv("historical_data.csv")

sentiment.columns = sentiment.columns.str.strip().str.lower()
trader.columns = trader.columns.str.strip().str.lower()

sentiment['date'] = pd.to_datetime(sentiment['date'], dayfirst=True)

trader['timestamp ist'] = pd.to_datetime(
    trader['timestamp ist'],
    dayfirst=True,
    errors='coerce'
)


trader = trader.dropna(subset=['timestamp ist'])

trader['date'] = trader['timestamp ist'].dt.date
sentiment['date'] = sentiment['date'].dt.date


sentiment.rename(columns={'classification': 'sentiment'}, inplace=True)
trader.rename(columns={'closed pnl': 'pnl'}, inplace=True)


df = pd.merge(trader, sentiment[['date', 'sentiment']], on='date', how='inner')


 
df['win'] = df['pnl'].apply(lambda x: 1 if x > 0 else 0)

avg_trade_size = df['size usd'].mean()


trades_per_day = df.groupby('date').size()


long_short = df['side'].value_counts(normalize=True)

print("Average Trade Size:", avg_trade_size)
print("\nLong/Short Ratio:\n", long_short)


plt.figure()
sns.boxplot(x='sentiment', y='pnl', data=df)
plt.xticks(rotation=45)
plt.title("PnL vs Sentiment")
plt.show()


plt.figure()
sns.barplot(x='sentiment', y='win', data=df)
plt.xticks(rotation=45)
plt.title("Win Rate vs Sentiment")
plt.show()


plt.figure()
sns.countplot(x='sentiment', data=df)
plt.xticks(rotation=45)
plt.title("Trades Count by Sentiment")
plt.show()


plt.figure()
sns.boxplot(x='sentiment', y='size usd', data=df)
plt.xticks(rotation=45)
plt.title("Trade Size vs Sentiment")
plt.show()


trade_counts = df['account'].value_counts()

df['trader_type'] = df['account'].apply(
    lambda x: 'Frequent' if trade_counts[x] > 10 else 'Infrequent'
)

df['performance'] = df['pnl'].apply(
    lambda x: 'Winner' if x > 0 else 'Loser'
)

print(df[['account', 'trader_type', 'performance']].head())


plt.figure()
sns.heatmap(df[['pnl','size usd']].corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()
