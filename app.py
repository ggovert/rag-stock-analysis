import streamlit as st
import os
from groq import Groq
from pinecone import Pinecone
#from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import requests
from datetime import datetime, timedelta



# Initialize pinecone
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pc = Pinecone(api_key=pinecone_api_key)
# Connect to pinecone database
pinecone_index = pc.Index("stocks2")


# Initialize GROQ
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# Load model globally
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Create huggingface embeddings
def get_huggingface_embeddings(text):
    return embedding_model.encode(text)

# Retrieve relevant documents from Pinecone
def perform_rag(query):
    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=10, include_metadata=True, namespace="stock-descriptions")

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    #Get top 3 article
    news_articles = fetch_top_news(query)


    augmented_query_with_news = (
    "<CONTEXT>\n" + "\n\n ---------- \n\n".join(contexts[:10]) +
"\n-------\n</CONTEXTS>" + "\n\n\n<RELATED NEWS ARTICLE>\n" +
    "\n-------\n".join(news_articles) + "\n-------\n</RELATED NEWS ARTICLE>"
"\n\n\nMY QUESTION:\n" + query
)
    # Modify the prompt below as need to improve the response quality

    system_prompt = """
    You are a financial expert specializing in stocks and the stock market.
    Provide clear, accurate, and well-researched answers to any stock-related questions.
    If relevant, include key details such as company names, sectors, market capitalization, and recent trends.
    Additionally, consider related companies that might be impacted by the question's context, including suppliers, competitors, or companies that have a direct supply-demand effect.
    Ensure your responses are concise, actionable, and easy to understand.
    """


    chat_completion = client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": augmented_query_with_news}
    ]
    )

    return chat_completion.choices[0].message.content




# Fetch top news using newsapi
def fetch_top_news(search_term):

    one_month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    url = "https://api.thenewsapi.com/v1/news/top"
    params = {
        'api_token': st.secrets["NEWS_API_KEY"],
        'locale': 'us',
        'limit': 3,
        'search': search_term,
        'language': "en",
        'published_after': one_month_ago,
        'sort': "relevance_score"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        news_data = response.json()
        news_articles = []
        for article in news_data['data']:
            news_articles.append(
                f"title: {article['title']}\n"
                f"source: {article['source']}\n"
                f"date: {article['published_at']}\n"
                f"description: {article['description']}\n"
                f"categories: {', '.join(article['categories'])}\n"
                f"link: {article['url']}\n"
            )

        return news_articles

    else:
        print(f"Failed to fetch news: {response.status_code}")
        return []



# Streamlit UI
def app():
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Choose a page", ["News Fetcher", "Chatbot"])


        if page == "News Fetcher":
                st.title("News Fetcher")
                st.write("Data is limited to the last 30 days.")
                search_term = st.text_input("Enter a company name:")
                if st.button("Fetch News"):
                        if search_term:
                                news_articles = fetch_top_news(search_term)
                                if news_articles:
                                        for idx, article in enumerate(news_articles):
                                                st.subheader(f"Article {idx + 1}")
                                                st.markdown(article)
                                                st.write("----")

                                else:
                                        st.write("No articles found.")
                        else:
                                st.write("Please enter a stock ticker.")

                        system_prompt = """
                                        You are an financial expert specializing in stocks and the stock market. 
                                        you will do sentiment analysis on each artticle given and give a summarize 
                                        based on the article given to you, whther the article is a good sign or not.
                                        make it really concise, short and easy to understand.
                                        """

                        
                        summary = client.chat.completions.create(
                                        model="llama-3.1-70b-versatile",
                                        messages=[
                                                {"role": "system", "content": system_prompt},
                                                {"role": "user", "content": ', '.join(news_articles)}
                                        ]
                                        )
                        st.subheader("Summary of the news:")
                        st.markdown(summary.choices[0].message.content)



        elif page == "Chatbot":
                st.title("Stock Chatbot")
                query = st.text_area("Ask me anything about stocks!")
                if st.button("Ask"):
                        st.markdown(perform_rag(query))



if __name__ == "__main__":
    app()