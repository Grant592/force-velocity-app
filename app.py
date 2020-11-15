from fv_profile import fvProfile
import dash_core_components as dcc
import dash_html_components as html
import os
import csv
import pandas as pd
import numpy as np
import dash
import dash_table
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import base64
import io
from collections import OrderedDict

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

def parse_csv(csv_file, filename):
    """Helper function to parse each individual csv file
    uploaded by user and process the data prior to graphing"""
    name, bw, date = filename.replace(".csv", '').split("-")
    label = f'{name}-{date}'
    content_type, content_string = csv_file.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')),skiprows=1 , sep=";") #header=None
    name = " ".join([x.title() for x in name.split("_")])
    df.columns = ['time', 'data', 'vel', 'lat', 'long']
    fv = fvProfile(df, name, float(bw))
    fv.smooth_data()
    fv.find_peaks()
    fv.extract_peaks()
    fv.extract_accel_phase()
    fv.model_sprint_data()
    fv.apply_calculations()
    fv.calculate_params()

    return label, fv.sprints, fv.sprint_dict


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    dcc.Store(id='memory-store'),
    html.H1("Force Velocity Profile Dashboard"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    dcc.Dropdown(
        id='dropdown-menu',
        multi=True
    ),
    html.Div(
        id='table1'
    ),
    html.Div(
        [
            dcc.Graph(
                id='force-speed-graph',
                className='six columns' 
            ),
            dcc.Graph(
                id='force-power-graph',
                className='six columns'
            ),
        ],
        className='row'
    ),
    html.Div(
        [
            dcc.Graph(
                id='power-velocity-graph',
                className='six columns' 
            ),
            dcc.Graph(
                id='horizontal-force-graph',
                className='six columns' 
            )
        ],
        className='row'
    )
])


@app.callback(
    Output('memory-store', 'data'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')])
def store_sprint_data(csv_files, filenames):
    master_dict = {}
    for csv_file, filename in zip(csv_files, filenames):
        label, sprint_data, sprint_dict = parse_csv(csv_file, filename)
        master_dict[label] = {}
        master_dict[label]['sprint_data'] = [df.to_dict(into=OrderedDict) for df in sprint_data]
        master_dict[label]['sprint_dict'] = sprint_dict

    return master_dict

@app.callback(
    Output('dropdown-menu', 'options'),
    [Input('memory-store', 'data')])
def create_dropdown(data):
    options = []
    for key in data.keys():
        name, date = key.split("-")
        name = " ".join(name.split("_")).title()
        sprints = [sprint for sprint in data[key]['sprint_dict'].keys()]
        for sprint in sprints:
            key_sprint = "-".join([key,sprint])
            options.append({'label':f'{name} {date} {sprint}', 'value':key_sprint})
    print(options)
    return options


@app.callback(
    Output('table1', 'children'),
    [Input('memory-store', 'data'),
    Input('dropdown-menu','value')])
def update_datatable(data, keys):
    df_list = []
    for key in data.keys():
        _,date = key.split("-")
        for sprint in data[key]['sprint_dict'].keys():
            data[key]['sprint_dict'][sprint]['date'] = date  
        dff = pd.DataFrame(data[key]['sprint_dict']).T.reset_index()
        dff = dff.rename(columns={'index':'sprint_no'})
        dff = dff[['name', 'date', 'sprint_no', 'bodyweight','F0', 'F0_kg','V0','Pmax', 'Pmax_kg', 'RF_max']].convert_dtypes()
        for col in dff.columns:
            if dff[col].dtype == 'float64':
                dff[col] = dff[col].round(2)
        df_list.append(dff)
    df = pd.concat(df_list, ignore_index=True)
    table = dash_table.DataTable(id='data-table',
                                 columns=[{'name':k,'id':k} for k in df.columns],
                                 data=df.to_dict('records'))
    return [html.H4('Sprint Data'),
            table]

@app.callback(
    Output('force-speed-graph','figure'),
    [Input('memory-store', 'data'),
     Input('dropdown-menu', 'value')])
def force_speed_graph(data, keys):
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.update_layout(title_text='Force & Speed Profile')
    fig.update_xaxes(
        title_text='Time (s)',
    )
    fig.update_yaxes(
        title_text='Velocity (m/s)',
        range=[0,10]
    )
    fig.update_yaxes(
        title_text='Acceleration (m/s/s)',
        secondary_y=True,
        range=[0,10]
    )
    for key in keys:
        key_list = key.split("-")
        date = key_list[1]
        key = "-".join(key_list[:2])
        sprint_id = int(key_list[2][-1])
        sprint = data[key]['sprint_data'][sprint_id]
        df = pd.DataFrame(sprint)
        fig.add_trace(
            go.Scatter(
                x = df['time'],
                y = df['vel'],
#                name=f'Vel- {idx} - {date}',
                name = f'Vel - {key}',
                mode='markers'
            )
        )
        fig.add_trace(
            go.Scatter(
                x = df['time'],
                y = df['predicted_velocity'],
                name='Predicted Vel'
            )
        )
        fig.add_trace(
            go.Scatter(
                x = df['time'],
                y = df['acceleration'],
                name='Acceleration'
            ),
            secondary_y=True
        )

    return fig


@app.callback(
    Output('force-power-graph','figure'),
    [Input('memory-store', 'data'),
     Input('dropdown-menu', 'value')])
def force_speed_graph(data, keys):
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.update_layout(title_text='Force & Power Profile')
    fig.update_xaxes(
        title_text='Time (s)',
    )
    fig.update_yaxes(
        title_text='Force (N/kg) and Speed (m/s)',
        range=[0,10]
    )
    fig.update_yaxes(
        title_text='Power (W/kg)',
        secondary_y=True,
        range=[0,20],
        dtick=4
    )
    for key in keys:
        key_list = key.split("-")
        date = key_list[1]
        key = "-".join(key_list[:2])
        sprint_id = int(key_list[2][-1])
        sprint = data[key]['sprint_data'][sprint_id]
        df = pd.DataFrame(sprint)
        fig.add_trace(
            go.Scatter(
                x = df['time'],
                y = df['predicted_velocity'],
                name=f'Vel - {key}',
            )
        )
        fig.add_trace(
            go.Scatter(
                x = df['time'],
                y = df['Ftot_kg'],
                name='Force/kg'
            )
        )
        fig.add_trace(
            go.Scatter(
                x = df['time'],
                y = df['PowerHzt_kg'],
                name='Power (W/kg)'
            ),
            secondary_y=True
        )

    return fig


@app.callback(
    Output('power-velocity-graph','figure'),
    [Input('memory-store', 'data'),
     Input('dropdown-menu', 'value')])
def force_speed_graph(data, keys):
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.update_layout(title_text='Field Sprint Force-Velocity-Power Profile')
    fig.update_xaxes(
        title_text='Time (s)',
    )
    fig.update_yaxes(
        title_text='Force (N/kg)',
        range=[0,8]
    )
    fig.update_yaxes(
        title_text='Power (W/kg)',
        secondary_y=True,
        range=[0,16],
        dtick=4
    )
    for key in keys:
        key_list = key.split("-")
        date = key_list[1]
        key = "-".join(key_list[:2])
        sprint_id = int(key_list[2][-1])
        sprint = data[key]['sprint_data'][sprint_id]
        df = pd.DataFrame(sprint)
        fig.add_trace(
            go.Scatter(
                x = df['predicted_velocity'],
                y = df['Ftot_kg'],
                name=f'Vel - {key}',
            )
        )
        fig.add_trace(
            go.Scatter(
                x = df['predicted_velocity'],
                y = df['PowerHzt_kg'],
                name='Power (W/kg)'
            ),
            secondary_y=True
        )

    return fig

@app.callback(
    Output('horizontal-force-graph','figure'),
    [Input('memory-store', 'data'),
     Input('dropdown-menu', 'value')])
def force_speed_graph(data, keys):
    fig = go.Figure()
    fig.update_layout(title_text='Horizontal Force Velocity Profile')
    fig.update_xaxes(
        title_text='Velocity (m/s))',
    )
    fig.update_yaxes(
        title_text='Ratio Force (%)',
    )
    for key in keys:
        key_list = key.split("-")
        date = key_list[1]
        key = "-".join(key_list[:2])
        sprint_id = int(key_list[2][-1])
        sprint = data[key]['sprint_data'][sprint_id]
        df = pd.DataFrame(sprint)
        fig.add_trace(
            go.Scatter(
                x = df['predicted_velocity'][df['time'] > 0.3],
                y = df['RF_perc'][df['time'] > 0.3],
                name=f'Ratio - {key}',
            )
        )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)


