import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder # Needed if LabelEncoder was used during training

# Load the trained model
@st.cache_resource # Cache the model loading for performance
def load_model():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- IMPORTANT: Re-create LabelEncoders or use pre-saved mappings if necessary ---
# In a real deployment, you would save the LabelEncoders (or their mappings)
# alongside your model, or ensure your app can correctly reproduce the encoding.
# For this demonstration, we'll assume the categorical inputs are handled carefully
# or manually mapped to their integer representations.
# For 'Gender', 'Education Level', and 'Job Title', we used LabelEncoder.
# You need to make sure the same encoding is applied to new inputs.

# Example mappings (these are derived from how the model was trained):
# Gender: Male -> 1, Female -> 0
# Education Level: Bachelor's -> 0, Master's -> 1, PhD -> 2
# Job Title: This is problematic for direct input. In a real app, you'd have a dropdown
# or text input that maps to the correct integer ID. For simplicity, we'll ask for an integer.

# Define the input features that the model expects
# Ensure these match the order and type of X_train used during training
# ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Input fields for user
age = st.slider('Age', 18, 65, 30)

gender_options = {'Male': 1, 'Female': 0}
gender_selection = st.selectbox('Gender', list(gender_options.keys()))
gender = gender_options[gender_selection]

education_options = {"Bachelor's": 0, "Master's": 1, "PhD": 2}
education_selection = st.selectbox('Education Level', list(education_options.keys()))
education_level = education_options[education_selection]

# For 'Job Title', direct integer input is not user-friendly.
# In a production app, you would have a dropdown populated with actual job titles
# and map them to their encoded integers using a saved mapping.
# For this example, we'll use a text input and expect the user to know the encoded value,
# or provide a simpler workaround.
# Let's provide a basic example, but highlight this limitation.
job_title_input = st.text_input('Job Title (Enter encoded integer value - e.g., 159 for Software Engineer)', '159')
try:
    job_title = int(job_title_input)
except ValueError:
    st.error('Please enter a valid integer for Job Title.')
    job_title = 0 # Default to avoid error

years_of_experience = st.slider('Years of Experience', 0.0, 40.0, 5.0, step=0.5)

if st.button('Predict Salary'):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Education Level': education_level,
        'Job Title': job_title,
        'Years of Experience': years_of_experience
    }])

    # Make prediction
    prediction = model.predict(input_data)

    st.success(f'Predicted Salary: ${prediction[0]:,.2f}')

st.markdown("""
--- 
**Note on Job Title Encoding:** In a real-world application, you would typically save the `LabelEncoder` (or its mappings) used during training and load it here to convert user-friendly job titles (e.g., 'Software Engineer') back into their numerical representations automatically. For this demo, we're asking for the pre-encoded integer value.
""")
