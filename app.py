"""
This scirpt allows the user to select a ground truth
"""
# DASH
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_table
import dash_daq as daq
import plotly.graph_objs as go

import time
import numpy as np
import argparse
import cv2
import ast
import json

from utils import GtSelection, hyper2rgb

parser = argparse.ArgumentParser(
    description="Select ground truth from a npy satellite / airborne image"
)

parser.add_argument(
    "--img_path", type=str, help='Path of the npy array'
)

args = parser.parse_args()

gt_selection = GtSelection(args.img_path)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, update_title=None, external_stylesheets=external_stylesheets)

app.layout = html.Div(id='container', children=[
    html.Div(id='classes', className='box', children=[
        html.H1('Classes'),
        html.Div(id='table', children=[
            dash_table.DataTable(
                data=gt_selection.set.to_dict('records'),
                style_cell={'textAlign': 'left'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                columns=[{'name': i, 'id': i} for i in gt_selection.set.columns]
            )
        ]),
        dcc.Interval(
                id='interval-component',
                interval=1*1000,
                n_intervals=0
            )]),
    html.Div(className='box', children =[
        html.H1('Add class'),
        html.Div(id='add', children =[
            daq.ColorPicker(
                id='color-picker',
                label='Select color',
                size=150,
                value=dict(rgb=dict(r=255, g=0, b=0, a=0))),
            html.Div(dcc.Input(id='label-input', className='item', size='10', type='text')),
            html.Button('Add a class', id='add-class', className='item', n_clicks=0)])
        ]),
    html.Div(id='select-gt', className='box', children=[
        html.H1('Select ground truth'),
        html.Button('Start', id='add-px', className='item', n_clicks=0),
        html.Button('Next', id='next', className='item', n_clicks=0),
        dcc.Store(id='timestamp', storage_type='session',data=None),
        dcc.Store(id='pt', storage_type='session'),
        html.Button('Reset', id='reset', className='item', n_clicks=0),
        html.Div(id='selecting'),
        html.Div(id='reseting')]),
    html.Div(className='box', children=[
        html.H1('RGB bands'),
        html.Div(className='slide', children=[
            dcc.Slider(id='slider-1', className='slider', min=1,max=gt_selection.n_bands,step=1,value=50),
            html.Div(id='band-1')]),
        html.Div(className='slide', children=[
            dcc.Slider(id='slider-2', className='slider', min=1,max=gt_selection.n_bands,step=1,value=30),
            html.Div(id='band-2')]),
        html.Div(className='slide', children=[
            dcc.Slider(id='slider-3', className='slider', min=1,max=gt_selection.n_bands,step=1,value=15),
            html.Div(id='band-3')]),
        dcc.Store(id='bands', storage_type='session')
        ]),
    html.Div(id='figures', className='box', children=[
        html.H1('Spectrum'),
        dcc.Graph(id='spectrum')])
    ])

@app.callback([Output('band-1', 'children'),
               Output('band-2', 'children'),
               Output('band-3', 'children')],
              [Input('slider-1', 'value'),
               Input('slider-2', 'value'),
               Input('slider-3', 'value')])
def update_bands(b1,b2,b3):
    rgb_bands = tuple((b1-1,b2-1,b3-1))
    gt_selection.rgb = hyper2rgb(gt_selection.HS_img,rgb_bands)
    gt_selection.rgb = gt_selection.rgb[:,:,::-1]
    gt_selection.RGB = cv2.UMat(gt_selection.rgb)
    return html.P('R: {}'.format(b1)), html.P('G: {}'.format(b2)), html.P('B: {}'.format(b3))

@app.callback([Output('table', 'children'), Output('pt','data')],
              [Input('interval-component', 'n_intervals')])
def update_class_table(n):
    set = gt_selection.set
    if len(set) < 1:
        raise PreventUpdate
    else:
        return html.Div([
            dash_table.DataTable(
                data=set.to_dict('records'),
                style_cell={'textAlign': 'left'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                columns=[{'name': i, 'id': i} for i in set.columns])
                    ]), gt_selection.pt

@app.callback(
     Output('label-input', 'value'),
    [Input('add-class', 'n_clicks_timestamp'),
     Input('color-picker', 'value')],
    [State('label-input', 'value')])
def update_label_input(ts, color, label):
    if ts is None or label == '':
        raise PreventUpdate
    else:
        set = gt_selection.set
        color = color['rgb']
        color = list(color.values())
        color = tuple(color)
        color = str(color)
        set.loc[len(set)] = [label,color,0]
        gt_selection.set = set
        return ''

@app.callback(
    Output('selecting','children'),
    Input('next','n_clicks_timestamp'))
def next(ts):
    if ts is None:
        raise PreventUpdate
    else:
        gt_selection.next = True

@app.callback(Output('reseting','children'),
    Input('reset','n_clicks_timestamp'))
def reset(ts):
    if ts is None:
        raise PreventUpdate
    else:
        gt_selection.reset = True

@app.callback(
    Output('timestamp', 'data'),
    Input('add-px', 'n_clicks_timestamp'))
def select_px(ts):
    if ts is None:
        raise PreventUpdate
    else:
        set = gt_selection.set
        labels = list(set['labels'])
        colors = list(set['colors'])
        n_class = len(colors)
        nb_px = gt_selection.select_gt(n_class, colors)
        for j,n in enumerate(nb_px):
            set.loc[j] = [labels[j], colors[j], n]
        gt_selection.set = set
        return ts

@app.callback(
    Output('spectrum', 'figure'),
    Input('pt','data'))
def plot_spectrum(pt):
    pt = gt_selection.pt
    if pt is None:
        fig = go.Figure(data=[go.Scatter(y=[0])])
        fig.layout.paper_bgcolor = '#F5F5F5'
        return fig
    else:
        fig = go.Figure(data=[go.Scatter(y=gt_selection.HS_img[pt[1],pt[0],:])])
        fig.layout.paper_bgcolor = '#F5F5F5'
        fig.update_xaxes(title='Spectral bands')
        fig.update_yaxes(title='Reflectance')
        return fig

if __name__ == '__main__':
    app.run_server(debug=True)
