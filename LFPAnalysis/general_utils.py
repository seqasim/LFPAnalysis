from random import random

from bokeh.layouts import row
from bokeh.models import ColumnDataSource, CustomJS, Button
from bokeh.events import ButtonClick
from bokeh.io import output_notebook, show, push_notebook
from bokeh.plotting import figure, output_file, show
import numpy as np

# UPDATE: deprecated since mne interactive plotting is working now. 

def bokeh_scroll_plot(timestamps, data, win_samples=1000, sub_sample=10): 
    """
    Code for visualizing and annotating data in bokeh. This was necessary because mne's interactive plot wasn't working in Jupyter notebooks on
    our headless Minerva server. This is a rudimentary function for approximating that functionality. 

    Note: You should sub-sample your timestamps and data to about 1000 Hz to ensure bokeh can plot all your points. 
    Save after every selection to output a text file of timestamps! 
    
    win_samples should be the length, in samples, that you want to see at a time when panning from left-to-right

    ex: bokeh_scroll_plot(raw['data'][1],
                  raw['data'][0][53, :], win_samples=30000, sub_sample=2)
    
    """
    output_notebook()

    win_range = timestamps[win_samples]
    s1 = ColumnDataSource(data=dict(x=timestamps, y=data))
    s1_subsampled = ColumnDataSource(data=dict(x=timestamps[0::sub_sample], y=data[0::sub_sample]))
    p1 = figure(width=800, height=300, tools=["box_select", "reset", "xpan"], title="Select Epochs", 
                x_range=(timestamps[0], win_range), y_range=(np.nanmin(data)*2, np.nanmax(data)*2))
    p1.circle('x', 'y', source=s1_subsampled, alpha=0)
    p1.line('x', 'y', source=s1, alpha=0.6)

    s1.selected.js_on_change('indices', CustomJS(args=dict(s1=s1_subsampled), code="""
            const inds = cb_obj.indices;
            const d1 = s1_subsampled.data;
            document.getElementsByTagName("button")[0].style.visibility = inds.length > 0 ? 'visible' : 'hidden';

        """)
    )
    
    savebutton = Button(label="Save", button_type="success")
    savebutton.js_on_event(ButtonClick, CustomJS(
        args=dict(source_data=s1_subsampled),
        code="""
            console.log('Saving!');
            var inds = source_data.selected.indices;
            var data = source_data.data;
            var out = "x, y\\n";
            for (let i = 0; i < inds.length; i++) {
                if (i === 0) {
                  out += data['x'][inds[i]] + "," + data['y'][inds[i]] + "\\n";
                } else if (i >= inds.length-1) {
                  out += data['x'][inds[i]] + "," + data['y'][inds[i]] + "\\n";
                }
            }
            var file = new Blob([out], {type: 'text/plain'});
            var elem = window.document.createElement('a');
            elem.href = window.URL.createObjectURL(file);
            elem.download = 'selected-data.txt';
            document.body.appendChild(elem);
            elem.click();
            document.body.removeChild(elem);
            """,
    ))

    layout = row(p1, savebutton)

    show(layout)