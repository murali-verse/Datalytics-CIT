# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🧼 Load the cleaned dataset
@st.cache_data
def load_data():
    df = pd.read_csv("D:\PYTHON\Project\sample_cleaned.csv")
    df['cleaned_content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    return df

# 🧠 TF-IDF Vectorization
@st.cache_resource
def compute_tfidf(df):
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['cleaned_content'])
    return tfidf_matrix

# 🔁 Function to get top similar articles
def get_similar_articles(article_index, tfidf_matrix, df, top_n=5):
    cosine_sim = cosine_similarity(tfidf_matrix[article_index], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    return similar_indices

# 🌐 Streamlit UI
def main():
    st.title("📚 Recommendation System")

    df = load_data()
    tfidf_matrix = compute_tfidf(df)

    st.write("Select an article to find similar ones based on title and content:")

    # Article selection by title
    titles = df['title'].fillna("Untitled").tolist()
    selected_title = st.selectbox("Choose an article title:", titles)

    if selected_title:
        selected_index = df[df['title'] == selected_title].index[0]

        top_n = st.slider("Number of similar articles to display", 1, 10, 5)
        similar_indices = get_similar_articles(selected_index, tfidf_matrix, df, top_n=top_n)

        st.subheader("🔍 Input Article:")
        st.write(df.iloc[selected_index]['title'])

        st.subheader("📚 You Might Also Like:")
        for i, idx in enumerate(similar_indices, 1):
            st.markdown(f"**{i}. {df.iloc[idx]['title']}**")
        print(df['tags'].head(10))

if __name__ == "__main__":
    main()
