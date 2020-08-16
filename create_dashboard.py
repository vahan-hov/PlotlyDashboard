"""please see a detailed explanation of my preprocessing and text normalization steps
    as well as some more statistical analysis in the notebook -
    https://colab.research.google.com/drive/1byNKl2Y_vzdOGunOS0a8cC6nz8WuMEX8?usp=sharing
"""
import re
import dash
import pandas as pd
import dash_table as dt
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

column2type = {'MONTH': 'datetime', 'SERVICE_CATEGORY': 'text', 'CLAIM_SPECIALTY': 'text', 'PAYER': 'text',
               'PAID_AMOUNT': 'numeric'}


def get_data():
    return pd.read_csv('Data/claims_test.csv')


def preprocess_month(df):
    df['MONTH'] = df['MONTH'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6])
    # convert to datetime and remove wrongly formatted dates like 2019-00
    df['MONTH'] = pd.to_datetime(df['MONTH'], format='%Y-%m', errors='coerce').dropna()
    return df


def is_same(item, same):
    original = item
    for replace_str in ['and', '&', '/', ' ']:
        item = item.replace(replace_str, '')

    if item in same:
        return same[item]
    return original


def get_almost_same(df):
    seen = set()
    same_dict = {}

    for unique in df.CLAIM_SPECIALTY.unique():
        original = unique

        for item in ['and', '&', '/', ' ']:
            unique = unique.replace(item, '')
        if unique in seen:
            same_dict[unique.lower()] = original.lower()
        seen.add(unique)

    return same_dict


def normalize_text(df):
    df = df.fillna('')

    text_columns = [col for col in df.columns if column2type[col] == 'text']
    for col in text_columns:
        df[col] = df[col].str.lower().apply(lambda x: re.sub('\t$', '', x))

    same_dict = get_almost_same(df)

    df['CLAIM_SPECIALTY'] = df['CLAIM_SPECIALTY'].apply(is_same, args=(same_dict,))

    similars = {"ob/gyn": "ob-gyn",
                "pulmonary disease": "pulmonary diseases",
                "other": "others",
                "physician's assistant": "physicians assistant",
                "infectious diseases": "infectious disease",
                "physician assistant": "physicians assistant",
                "mdwife": "midwife",
                "nurse practitioner": "nurse practitioners",
                "ambulatory surgical center": "ambulatory surgical centers"}

    df['CLAIM_SPECIALTY'] = df['CLAIM_SPECIALTY'].replace(similars)
    return df


def get_mean_plot_figures(df):
    figures = []
    plot_columns = ['SERVICE_CATEGORY', 'CLAIM_SPECIALTY', 'PAYER']
    for col in plot_columns:
        if col == 'CLAIM_SPECIALTY':
            fig = px.bar(df.groupby(col).PAID_AMOUNT.mean().reset_index(), x=col, y='PAID_AMOUNT',
                         title=f"MEAN PAID AMOUNT BY {col}", height=1000)
        else:
            fig = px.bar(df.groupby(col).PAID_AMOUNT.mean().reset_index(), x=col, y='PAID_AMOUNT',
                         title=f"MEAN PAID AMOUNT BY {col}")
        figures.append(fig)
    return figures


def get_paid_amount_over_time_figure(df):
    df_months = df.groupby('MONTH').mean().reset_index()
    return px.line(df_months, x="MONTH", y="PAID_AMOUNT", title='mean PAID AMOUNT over time')


def main():
    df = get_data()
    df = preprocess_month(df)
    df = normalize_text(df)
    mean_plot_figures = get_mean_plot_figures(df)
    paid_amount_figure = get_paid_amount_over_time_figure(df)

    app = dash.Dash(__name__, prevent_initial_callbacks=True)
    app.layout = html.Div([
        html.Div(children=[dt.DataTable(
            id='table',
            columns=[{"name": col, "id": col, "type": column2type[col]} for col in df.columns],
            data=df.to_dict('records'),
            filter_action='native',
            row_selectable='multi',
            column_selectable='multi')]),

        html.Div([dcc.Graph(figure=fig) for fig in mean_plot_figures]),

        html.Div(dcc.Graph(figure=paid_amount_figure))
    ])

    @app.callback(Output('table', 'data'),
                  [Input('table', 'selected_rows')])
    def update_graphs(selected_rows):
        return df.iloc[selected_rows, :].to_dict('records')

    app.run_server(debug=True)


if __name__ == '__main__':
    main()
