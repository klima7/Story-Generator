import streamlit as st
from lstm import LstmTextGenerator


@st.cache_resource
def get_model():
    ckpt_path = '../lstm/logs/version_23/checkpoints/epoch=57-step=187000.ckpt'
    model = LstmTextGenerator.load_from_checkpoint(ckpt_path)
    return model
    

st.title('LSTM Story generator')

prompt = st.text_area('Prompt', value='Dawno, dawno temu, był sobie')
temperature = st.slider('Temperature', 0.0, 1.0, 0.3, 0.01)
length = st.number_input('Length', min_value=10, max_value=2_000, value=300, step=100)

if st.button('Generate'):
    model = get_model()
    
    bar = st.progress(0.0)
    
    generated = model.generate(
        prompt,
        temperature=temperature,
        length=length,
        progress_callback=lambda n: bar.progress(n/length)
    )
    st.write(generated)
