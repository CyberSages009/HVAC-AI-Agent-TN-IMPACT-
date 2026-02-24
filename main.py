import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HVAC AI Agent", layout="wide")

st.title("HVAC AI Agent")
st.write("Your Streamlit environment is working.")

st.subheader("Sample Data")
df = pd.DataFrame(
    {
        "Hour": list(range(1, 13)),
        "Temperature_C": np.random.normal(loc=24, scale=2, size=12).round(2),
        "Humidity_%": np.random.normal(loc=55, scale=8, size=12).round(2),
    }
)

st.dataframe(df, use_container_width=True)
st.line_chart(df.set_index("Hour")[["Temperature_C", "Humidity_%"]])
