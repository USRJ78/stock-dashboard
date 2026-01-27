File "/mount/src/stock-dashboard/streamlit_app.py.py", line 149, in <module>
    st.line_chart(prices)
    ~~~~~~~~~~~~~^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/metrics_util.py", line 532, in wrapped_func
    result = non_optional_func(*args, **kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/vega_charts.py", line 877, in line_chart
    chart, add_rows_metadata = generate_chart(
                               ~~~~~~~~~~~~~~^
        chart_type=ChartType.LINE,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<9 lines>...
        use_container_width=(width == "stretch"),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/lib/built_in_chart_utils.py", line 216, in generate_chart
    df, x_column, y_column, color_column, size_column, sort_column = _prep_data(
                                                                     ~~~~~~~~~~^
        df, x_column, y_column_list, color_column, size_column, sort_column
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/lib/built_in_chart_utils.py", line 492, in _prep_data
    selected_data = _drop_unused_columns(
        df, x_column, color_column, size_column, sort_column, *y_column_list
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/lib/built_in_chart_utils.py", line 670, in _drop_unused_columns
    return df[keep]
           ~~^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/core/frame.py", line 4119, in __getitem__
    indexer = self.columns._get_indexer_strict(key, "columns")[1]
              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/core/indexes/multi.py", line 2782, in _get_indexer_strict
    self._raise_if_missing(key, indexer, axis_name)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/pandas/core/indexes/multi.py", line 2800, in _raise_if_missing
    raise KeyError(f"{keyarr[cmask]} not in index")
