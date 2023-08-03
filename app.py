from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from src.main import predict, start_model_training

st.set_page_config('Prevention System', 'random')
st.markdown(
    '<h2 style="color: orange;" align=center>'
    'Money Laundering Prevention System'
    '</h2>',
    True,
)


@dataclass
class BaseDF:
    sourceid: int
    destinationid: int
    amountofmoney: int
    month: int
    typeofaction: str
    typeoffraud: str

    def to_dict(self):
        return {
            'sourceid': sourceid,
            'destinationid': destinationid,
            'amountofmoney': amountofmoney,
            'month': month,
            'typeofaction': typeofaction,
            'typeoffraud': typeoffraud,
        }


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
# Global Variables
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
msg = st.empty()
base = None

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
# Sidebar
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
with st.sidebar:
    option = st.selectbox('Prediction Type',
                          [
                              'Prediction from Form', #'Batch Prediction'
                          ])

# --- Train model button --- #
with st.spinner('Model is training...'):
    if st.button('Train Model', use_container_width=True):
        # if connection with MONGODB is not established then
        # data is fetched from `ingestion_data_path` argument.
        start_model_training(Path('data/base_data.csv'))
        st.balloons()

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
# Two different form declaration
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
if option == 'Prediction from Form':
    with st.form('prediction-from-form'):
        sourceid = int(st.number_input('Source ID',
                                       format='%d', value=30105))
        destinationid = int(st.number_input('Destination ID',
                                            format='%d', value=8692))
        amountofmoney = int(st.number_input('Amount of Money',
                                            format='%d', value=494528))
        month = int(st.number_input('Month of transaction',
                                    format='%d', value=5))
        typeofaction = str(st.selectbox('Type of Action',
                                        ['cash-in', 'transfer']))
        typeoffraud = str(st.selectbox('Type of Fraud',
                                       ['type1', 'type2', 'type3', 'none']))

        if st.form_submit_button():
            base = BaseDF(sourceid, destinationid, amountofmoney,
                          month, typeofaction, typeoffraud)
else:
    with st.form('batch-prediction'):
        upload = st.file_uploader(label='Upload CSV file', type='csv')

        if st.form_submit_button():
            if upload is None:
                raise FileNotFoundError('File does not uploaded.')
            base = pd.read_csv(upload)

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
# Process after getting the `base` DataFrame
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- #
if base is not None and isinstance(base, BaseDF):
    df = pd.DataFrame([base.to_dict()])
    try:
        _, prediction = predict(df)
    except FileNotFoundError:
        msg.error('Model is trained yet. Please train model first.',
                  icon='🔥')
        st.stop()
    else:
        result, color = (('Fraud', 'red') if prediction == 1
                         else ('Not Fraud', 'green'))
        st.subheader(f':{color}[The entry is {result}.]')
elif base is not None and isinstance(base, pd.DataFrame):
    df = base
    try:
        pred_df, _ = predict(df)
    except FileNotFoundError:
        msg.error('Model is trained yet. Please train model first.',
                  icon='🤖')
        st.stop()
    else:
        st.balloons()
        msg.success('Download the predicted data file.')
        st.download_button(
            label='Download Prediction DataFrame',
            data=pred_df.to_csv(index=False),
            file_name='Money-Laundering-Prediction.csv',
            mime='csv',
            use_container_width=True,
        )
else:
    st.markdown('<h3 align=center>Submit the form.</h3>', True)
