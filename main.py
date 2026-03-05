import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('Driver_Drowsiness_3000.csv')
X = df[['EAR', 'MAR']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

x_min, x_max = X['EAR'].min() - 0.05, X['EAR'].max() + 0.05
y_min, y_max = X['MAR'].min() - 0.05, X['MAR'].max() + 0.05

eixo_x = np.arange(x_min, x_max, 0.01)
eixo_y = np.arange(y_min, y_max, 0.01)
xx, yy = np.meshgrid(eixo_x, eixo_y)

Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#exibindo a figura
fig1 = go.Figure()

fig1.add_trace(go.Contour(
    x=eixo_x, y=eixo_y, z=Z,
    colorscale='RdBu', 
    opacity=0.3,
    showscale=False,
    hoverinfo='skip'
))

df_acordado = df[df['Target'] == 0]
fig1.add_trace(go.Scatter(
    x=df_acordado['EAR'], y=df_acordado['MAR'],
    mode='markers', name='Acordado (0)',
    marker=dict(color='blue', line=dict(color='white', width=1))
))

df_sonolento = df[df['Target'] == 1]
fig1.add_trace(go.Scatter(
    x=df_sonolento['EAR'], y=df_sonolento['MAR'],
    mode='markers', name='Sonolento (1)',
    marker=dict(color='red', line=dict(color='white', width=1))
))

fig1.update_layout(
    title='Fronteira de Decisão - Regressão Logística',
    xaxis_title='EAR (Fechamento do Olho)',
    yaxis_title='MAR (Abertura da Boca)',
    width=800, height=600
)

fig1.show()

#fazendo predição nos teste e calculando a matriz confusão
y_pred = modelo.predict(X_test)
matriz = confusion_matrix(y_test, y_pred)

labels = ['Acordado (0)', 'Sonolento (1)']

fig2 = px.imshow(
    matriz, 
    text_auto=True, 
    color_continuous_scale='Blues',
    x=labels, 
    y=labels,
    labels=dict(x="Previsão do Modelo", y="Realidade")
)

fig2.update_layout(title='Matriz de Confusão', width=600, height=500)
fig2.show()