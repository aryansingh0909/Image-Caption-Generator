import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for the generation process
    in_text = 'startseq'
    # iterate over the max length of the sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get the index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating the next word
        in_text += " " + word
        # stop if we reach the end tag
        if word == 'endseq':
            break

    return in_text


def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc.,
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + \
                " ".join([word for word in caption.split()
                          if len(word) > 1]) + ' endseq'
            captions[i] = caption

def remove_tags(caption):
    start_tag = 'startseq'
    end_tag = 'endseq'
    
    # Remove start tag
    if caption.startswith(start_tag):
        caption = caption[len(start_tag):].strip()
    
    # Remove end tag
    if caption.endswith(end_tag):
        caption = caption[:-len(end_tag)].strip()
    
    return caption


def main():
    st.title("Image Caption Generator")
    st.markdown("---")
    st.subheader('Created by:')
    st.markdown('[Ankita Anand](https://www.linkedin.com/in/ankita-anand-75286b218/)')
    st.markdown('[Aryan Singh](https://www.linkedin.com/in/aryan-singh0909/)')

    # Display file uploader
    uploaded_file = st.file_uploader("Upload an image to generate the caption:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Read the image file and convert it to array
        image = load_img(uploaded_file, target_size=(224, 224))
        # convert image pixels to numpy array
        image_array = img_to_array(image)
        # Reshape the image data for the model
        image = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
        # preprocess image for VGG
        image = preprocess_input(image)
        # extract features
        vgg_model = VGG16()
        # restructure the model
        vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
        feature = vgg_model.predict(image, verbose=0)
        # Load the model
        model = load_model("best_model.h5")

        # Load the captions
        with open('captions.txt', 'r') as f:
            next(f)
            captions_doc = f.read()
        # create mapping of image to captions
        mapping = {}
        # process lines
        for line in captions_doc.split('\n'):
            # split the line by comma(,)
            tokens = line.split(',')
            if len(line) < 2:
                continue
            image_id, caption = tokens[0], tokens[1:]
            # remove extension from image ID
            image_id = image_id.split('.')[0]
            # convert caption list to string
            caption = " ".join(caption)
            # create list if needed
            if image_id not in mapping:
                mapping[image_id] = []
            # store the caption
            mapping[image_id].append(caption)
        # preprocess the text
        clean(mapping)
        all_captions = []
        for key in mapping:
            for caption in mapping[key]:
                all_captions.append(caption)
        # tokenize the text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_captions)
        vocab_size = len(tokenizer.word_index) + 1
        # get maximum length of the caption available
        max_length = max(len(caption.split()) for caption in all_captions)
        max_length
        # predict caption from the trained model
        with st.spinner("Generating caption..."):
            caption = predict_caption(model, feature, tokenizer, max_length)
            caption = remove_tags(caption)
        # Display the generated caption
        st.success('Caption generated!')
        st.subheader("Generated Caption:")
        st.write(caption)
        st.markdown("---")


if __name__ == "__main__":
    main()
