import tensorflow as tf
import requests
from bs4 import BeautifulSoup
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from transformers import TFT5ForConditionalGeneration, T5Tokenizer

# Load the T5 model and tokenizer
model_name = "t5-base"
model = TFT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512)


# Function to preprocess the image from URL
def preprocess_image_from_url(image_url):
    response = requests.get(image_url)
    image = tf.image.decode_jpeg(response.content, channels=3)
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image

# Function to preprocess the uploaded image
def preprocess_uploaded_image(uploaded_file):
    image = tf.image.decode_jpeg(uploaded_file.read(), channels=3)
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image

# Function to generate a caption for the image
def generate_caption(image):
    inputs = tokenizer(image, return_tensors="tf", padding=True, truncation=True, max_length=512)
    caption = model.generate(**inputs)
    caption = tokenizer.batch_decode(caption, skip_special_tokens=True)
    return caption[0]

# Function to scrape poem from the website based on the caption
def scrape_poem_from_website(caption):
    payload = {
        "input_text": caption,
        "button": "Generate Poem"
    }
    response = requests.post("https://www.aipoemgenerator.org", data=payload)
    soup = BeautifulSoup(response.text, 'html.parser')
    poem = soup.find('textarea', class_='block w-full rounded-lg border disabled:cursor-not-allowed disabled:opacity-50 bg-gray-50 border-gray-300 text-gray-900 focus:border-blue-500 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:placeholder-gray-400 dark:focus:border-blue-500 dark:focus:ring-blue-500 w-full', id='message').text.strip()
    return poem

# Streamlit app code
def main():
    st.title("Auto Poem Generator")

    # Generate poem from image URL
    st.subheader("Generate Poem from Image URL")
    image_url = st.text_input("Enter the URL of an image")
    if st.button("Generate Poem"):
        image = preprocess_image_from_url(image_url)
        caption = generate_caption(image)
        poem = scrape_poem_from_website(caption)
        st.subheader("Generated Poem:")
        st.write(poem)

    # Generate poem from uploaded image
    st.subheader("Generate Poem from Uploaded Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = preprocess_uploaded_image(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Generate Poem"):
            caption = generate_caption(image)
            poem = scrape_poem_from_website(caption)
            st.subheader("Generated Poem:")
            st.write(poem)

if __name__ == '__main__':
    nltk.download('punkt')  # Download the tokenizer data
    main()
