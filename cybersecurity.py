import pandas as pd
import plotly.express as px

df = pd.read_csv('cybersecurity.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.groupby(df["timestamp"].dt.date).size().reset_index(name="attacks")
fig = px.line(
    df,
    x="timestamp",
    y="attacks",
    title="Number of Attacks Over Time"
)

fig.show()
