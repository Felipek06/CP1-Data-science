import pandas as pd
import plotly.express as px

df_btc = pd.read_csv('bitcoin.csv')
df_btc['Date'] = pd.to_datetime(df_btc['Date'])
df_btc = df_btc.sort_values('Date')

fig = px.line(
    df_btc, 
    x='Date', 
    y='Close', 
    title='Histórico de Preço do Bitcoin (USD)',
    labels={'Date': 'Data', 'Close': 'Preço em Dólar (USD)'},
    log_y=True,
    template='plotly_dark' # Um tema escuro fica bem legal para dados financeiros
)

fig.update_traces(line=dict(width=1.5, color='#f2a900'))

fig.update_layout(
    yaxis=dict(
        tickformat="$",
        gridcolor='rgba(255, 255, 255, 0.1)'
    ),
    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
    hovermode="x unified"
)

fig.update_xaxes(rangeslider_visible=True)

fig.show()
