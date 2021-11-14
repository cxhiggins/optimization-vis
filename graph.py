#!/usr/bin/env python

import os
import plotly.graph_objects as go

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
import numpy as np
import math

# ---------- Graph generation functions ----------

# Returns a surface contour plot given arrays X, Y, and Z
def surface(X, Y, Z):
    return go.Surface(
            contours = {
                "z": {"show": True}
            },
            x = X,
            y = Y,
            z = Z,
            opacity = 0.8,
            hoverinfo="x+y+z"
        )

# Returns a scatter plot with the first x points
def scatter(i, ptsx, ptsy, ptsz):
    return go.Scatter3d(
        x=ptsx[0:i],
        y=ptsy[0:i],
        z=ptsz[0:i],
        marker={
            "size": 5,
            "color": 'white'
        },
        line={
            "color": 'white',
            "width": 2
        },
        text=labels,
        hoverinfo="x+y+z+text"
    )

# ---------- Plot 2 ----------

def f(x, y):
    return 1/2 * (math.pow(x+y,4)+math.pow(x-y,4))

def df(x, y):
    a = math.pow(x+y, 3)
    b = math.pow(x-y, 3)
    return 2*np.array([a+b, a-b])

# Set starting point for minimization
ptsx = [2] # x0
ptsy = [1] # y0
ptsz = [f(ptsx[0],ptsy[0])]
labels = ['x0']

maxsteps = 10    # Max num iterations
a = 1            # Initial step length of 1
x = 2; y = 1
sigma = 10**(-2) # Armijo constant
p = 1/2          # Reduction factor for backtracking

# Gradient descent with backtracking
for i in range(maxsteps):
    grad = df(x,y)
    lastpt = np.array([x,y])
    lastdot = np.dot([x,y],[x,y])
    
    [x,y] = lastpt - a * grad
    z = f(x,y)



    # Backtrack until Armijo rule satisfied
    j = 0
    while z >= ptsz[i] - sigma * a * lastdot:
        if (j == 20):
            print("Armijo rule is not satisfied after 20 steps")
            break
        a *= p
        [x,y] = lastpt - a * grad
        z = f(x,y)
        j += 1

    # Set initial step length for next iter so amt of descent remains constant
    a *= lastdot / np.dot([x,y],[x,y]) 
    
    ptsx += [x]
    ptsy += [y]
    ptsz += [z]
    labels += ['x'+str(i+1)]
    # print("("+str(x)+", "+str(y)+")")

gran = 20
xyrange = [min(ptsx+ptsy)-2,max(ptsx+ptsy)+2]
step = (xyrange[1] - xyrange[0])/gran
X = np.arange(xyrange[0], xyrange[1]+step, step)
Y = X
Z = np.zeros((gran+1,gran+1))

for i in range(gran+1):
    for j in range(gran+1):
        Z[i][j] = f(X[i], Y[j])

surface1 = surface(X, Y, Z)

layout = {
    'title': 'f(x,y) = 1/2((x+y)^4+(x-y)^4)',
    'height': 600,
    'scene': {
        "xaxis": {"nticks": 10},
        "yaxis": {"nticks": 10},
        "zaxis": {"nticks": 10},
        #'camera_eye': {"x": 0, "y": -1, "z": 0.5},
        "aspectratio": {"x": 1, "y": 1, "z": 1}
    }
}

fig = go.Figure(data=[
    surface1,
    scatter(5, ptsx, ptsy, ptsz)
], layout=layout)

mytable = dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in ["Iteration", "x", "f(x)"]],
    data=[{"Iteration": i, "x": "("+str(ptsx[i])+", "+str(ptsy[i])+")", "f(x)": str(ptsz[i])} for i in range(maxsteps+1)],
)

# ---------- Plot 2 ----------
def f2(x, y):
    return math.pow(x,2)+math.pow(y,2)+math.exp(2*x+y)

def df2(x, y):
    a = math.exp(2*x+y)
    k = 1 / (4 + 10*a)
    return k * np.array([4*x+2*a*(x-2*y+2), 4*y+2*a*(4*y-2*x+1)])

# Set starting point for minimization
ptsx2 = [0] # x0
ptsy2 = [0] # y0
ptsz2 = [f2(ptsx2[0],ptsy2[0])]
labels2 = ['x0']

maxsteps2 = 10    # Max num iterations
x = ptsx2[0]; y = ptsy2[0]

# Newton's method descent
for i in range(maxsteps):    
    [x,y] = np.array([x,y]) - df2(x,y) # Step length fixed to 1
    
    ptsx2 += [x]
    ptsy2 += [y]
    ptsz2 += [f2(x,y)]
    labels2 += ['x'+str(i+1)]

# gran = 20 
xyrange = [min(ptsx2+ptsy2)-2,max(ptsx2+ptsy2)+2]
step = (xyrange[1] - xyrange[0])/gran
X = np.arange(xyrange[0], xyrange[1]+step, step)
Y = X
Z = np.zeros((gran+1,gran+1))

for i in range(gran+1):
    for j in range(gran+1):
        Z[i][j] = f2(X[i], Y[j])

surface2 = surface(X, Y, Z)

layout2 = {
    'title': 'Find the point on the surface z = exp(x+y/2) that is closest to the origin',
    'height': 600,
    'scene': {
        "xaxis": {"nticks": 10},
        "yaxis": {"nticks": 10},
        "zaxis": {"nticks": 10},
        #'camera_eye': {"x": 0, "y": -1, "z": 0.5},
        "aspectratio": {"x": 1, "y": 1, "z": 1}
    }
}

fig2 = go.Figure(data=[
    surface2,
    scatter(5, ptsx2, ptsy2, ptsz2)
], layout=layout2)

mytable2 = dash_table.DataTable(
    id='table2',
    columns=[{"name": i, "id": i} for i in ["Iteration", "x", "f(x)"]],
    data=[{"Iteration": i, "x": "("+str(ptsx2[i])+", "+str(ptsy2[i])+")", "f(x)": str(ptsz2[i])} for i in range(maxsteps+1)],
)

##fig.show()

# Uses Dash as an alternative to plotly's fig.show()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Steepest Descent"

server = app.server

app.layout = html.Div([
    html.Div(
        [html.H1("Continuous Math - Chapter 6")],
        style={'textAlign': "center", "margin-top": '4%'}
    ),
    html.Div(
        [html.H2("Steepest Descent")],
        style={'textAlign': "center", "color":"gray"}
    ),
    html.Div(
        className="row",
        children=[
            html.Div(
                className="five columns",
                style={"margin-left":"2%"},
                children=[
                    html.H3("Number of Iterations"),
                    html.Div(
                        style={"width":"250px", "margin-bottom":"50px"},
                        children=[
                            dcc.Slider(
                                id='my-slider',
                                min=1,
                                max=maxsteps,
                                step=1,
                                value=5,
                                marks={i:{'label': str(i)} for i in range(1, maxsteps+1, 1)}
                            )
                        ]
                    ),
                    html.Div(mytable)]
            ),
            html.Div(
                children=[dcc.Graph(id='my-graph', figure=fig)],
                style={"margin-left":"0"},
                className="seven columns"
            )
        ]
    ),
    html.Div(
        [html.H2("Newton's Method")],
        style={'textAlign': "center", "color":"gray"}
    ),
    html.Div(
        className="row",
        children=[
            html.Div(
                className="five columns",
                style={"margin-left":"2%"},
                children=[
                    html.H3("Number of Iterations"),
                    html.Div(
                        style={"width":"250px", "margin-bottom":"50px"},
                        children=[
                            dcc.Slider(
                                id='my-slider2',
                                min=1,
                                max=maxsteps,
                                step=1,
                                value=5,
                                marks={i:{'label': str(i)} for i in range(1, maxsteps+1, 1)}
                            )
                        ]
                    ),
                    html.Div(mytable2)]
            ),
            html.Div(
                children=[dcc.Graph(id='my-graph2', figure=fig2)],
                style={"margin-left":"0"},
                className="seven columns"
            )
        ]
    )
])

@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('my-slider', 'value')])
def update_output(value):
    return go.Figure(data=[
            surface1,
            scatter(value, ptsx, ptsy, ptsz)
        ], layout=layout)

@app.callback(
    dash.dependencies.Output('my-graph2', 'figure'),
    [dash.dependencies.Input('my-slider2', 'value')])
def update_output(value):
    return go.Figure(data=[
            surface2,
            scatter(value, ptsx2, ptsy2, ptsz2)
        ], layout=layout2)


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
