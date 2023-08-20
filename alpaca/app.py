from transformers import pipeline
import streamlit as st


@st.cache_resource
def get_model_and_tokenizer():
    from alpaca import model, tokenizer
    return model, tokenizer
    

st.title('Alpaca Story generator')

prompt = st.text_area('Prompt', value='Dawno, dawno temu, był sobie')
temperature = st.slider('Temperature', 0.01, 1.0, 0.3, 0.01)
top_p = st.slider('Top P', 0.01, 1.0, 0.2, 0.01)
top_k = st.slider('Top K', 1, 300, 50, 1)
length = st.number_input('Length', min_value=10, max_value=2_000, value=300, step=100)

if st.button('Generate'):
    model, tokenizer = get_model_and_tokenizer()
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generated = generator(
        prompt,
        max_new_tokens=length,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=1,
        no_repeat_ngram_size=6,
        repetition_penalty=1.0,
        remove_invalid_values=True,
        top_k=top_k,
        top_p=top_p,
    )[0]['generated_text']
    st.write(generated)
