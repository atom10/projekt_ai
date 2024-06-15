import pandas as pd
import numpy as np

def generate_table_html(data, num_rows=None):
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    total_rows = len(data)

    if num_rows is None or num_rows >= total_rows:
        num_top_rows = total_rows
        num_bottom_rows = 0
    else:
        num_top_rows = num_rows // 2
        num_bottom_rows = num_rows - num_top_rows

    html = '<table class="table">'
    html += '<thead>'
    html += '<tr>'
    for column in data.columns:
        html += f'<th>{column}</th>'
    html += '</tr>'
    html += '</thead>'
    html += '<tbody>'

    for index, row in data.head(num_top_rows).iterrows():
        html += '<tr>'
        for value in row:
            html += f'<td>{value}</td>'
        html += '</tr>'

    if num_bottom_rows > 0:
        html += '<tr><td colspan="{}">...</td></tr>'.format(len(data.columns))

        for index, row in data.tail(num_bottom_rows).iterrows():
            html += '<tr>'
            for value in row:
                html += f'<td>{value}</td>'
            html += '</tr>'

    html += '</tbody>'
    html += '</table>'
    return html

def generate_metric_paragraph(label, value):
    return f'<div class="metric"><p>{label}</p><strong>{value}</strong></div>'

def generate_image(img_path):
    return f'<img src="{img_path}">'