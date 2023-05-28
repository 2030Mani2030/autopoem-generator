import requests
from io import BytesIO
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
from bs4 import BeautifulSoup
import base64

# Load the CLIP model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

# Create a cache object for storing generated poems
cache = {}

# Function to preprocess the image from URL
def preprocess_image_from_url(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = image.resize((224, 224))
    image = transforms.ToTensor()(image).unsqueeze(0)
    image = image.to(device)
    return image

# Function to preprocess the uploaded image
def preprocess_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    image = transforms.ToTensor()(image).unsqueeze(0)
    image = image.to(device)
    return image

# Function to generate a caption for the image
def generate_caption(image):
    inputs = processor(text=None, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probabilities = logits_per_image.softmax(dim=1)
        caption_index = torch.argmax(probabilities)
    caption = processor.tokenizer.decode(caption_index)
    return caption

# Function to scrape poem from the website based on the caption
def scrape_poem_from_website(caption):
    # Check if the poem has already been cached
    poem = cache.get(caption)
    if poem:
        return poem

    payload = {
        "input_text": caption,
        "button": "Generate Poem"
    }
    response = requests.post("https://www.aipoemgenerator.org", data=payload)
    soup = BeautifulSoup(response.text, 'html.parser')
    poem = soup.find('textarea', class_='block w-full rounded-lg border disabled:cursor-not-allowed disabled:opacity-50 bg-gray-50 border-gray-300 text-gray-900 focus:border-blue-500 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:placeholder-gray-400 dark:focus:border-blue-500 dark:focus:ring-blue-500 w-full', id='message').text.strip()

    # Cache the poem for future use
    cache[caption] = poem

    return poem

# Streamlit app code
def main():
    st.set_page_config(layout="wide")

    # Set theme mode (light/dark)
    theme_mode = st.sidebar.radio("Theme Mode", ("Light", "Dark"))
    if theme_mode == "Dark":
        st.markdown(
            """
            <style>
            .stApp {
                color: white;
                background-color: #121212;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Generate poem from image URL
    st.subheader("Generate Poem from Image URL")
    image_url = st.text_input("Enter the URL of an image", key="url_input")
    if st.button("Generate Poem", key="url_button"):
        try:
            image = preprocess_image_from_url(image_url)
            caption = generate_caption(image)
            poem = scrape_poem_from_website(caption)

            # Open the poem in a new tab
            encoded_image = base64.b64encode(requests.get(image_url).content).decode('utf-8')
            st.markdown(
                f"<a href='data:text/html;charset=utf-8,<html><body><img src=\"{image_url}\" alt=\"Image\" width=\"300\"><h2>Caption: {caption}</h2><p>Poem: {poem}</p></body></html>' target='_blank'>Open Poem in New Tab</a>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error("Error occurred during processing. Please try again.")

    # Generate poem from uploaded image
    st.subheader("Generate Poem from Uploaded Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="file_input")
    if st.button("Generate Poem", key="file_button"):
        try:
            image = preprocess_uploaded_image(uploaded_file)
            caption = generate_caption(image)
            poem = scrape_poem_from_website(caption)

            # Open the poem in a new tab
            encoded_image = base64.b64encode(uploaded_file.read()).decode('utf-8')
            st.markdown(
                f"<a href='data:text/html;charset=utf-8,<html><body><img src=\"data:image/png;base64,{encoded_image}\" alt=\"Image\" width=\"300\"><h2>Caption: {caption}</h2><p>Poem: {poem}</p></body></html>' target='_blank'>Open Poem in New Tab</a>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error("Error occurred during processing. Please try again.")

if __name__ == "__main__":
    main()
