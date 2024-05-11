import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from groq import Groq
import time

def get_transcript_text(video_url):
    parsed_url = urlparse(video_url)
    video_id = parse_qs(parsed_url.query).get("v", [None])[0]
    if video_id:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join(line["text"] for line in transcript)
        return transcript_text
    else:
        return "Invalid YouTube video URL"

def chunk_transcript(transcript_text):
    length = len(transcript_text)
    size = length // 4
    return [transcript_text[i:i+size] for i in range(0, length, size)]

def summarize_chunks(chunks, api_key):
    client = Groq(api_key=api_key)
    summaries = []
    for chunk in chunks:
        response = client.chat.completions.create(
            model='llama3-8b-8192',
            messages=[{
                "role": "system",
                "content": "Summarize this information in detail in bullet points - we will be combining it with additional information later to create more comprehensive content."
            }, {
                "role": "user",
                "content": chunk
            }],
            temperature=0.85,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None
        )
        # Ensure all responses are non-None before concatenation
        summary = "".join([resp.choices[0].delta.content for resp in response if resp.choices[0].delta.content is not None])
        summaries.append(summary)
        time.sleep(1)  # Pause to manage API rate limits and server load
    return summaries

def generate_blog_post(summaries, api_key):
    combined_summary = " ".join([s for s in summaries if s])  # Ensure no None summaries are included
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model='llama3-70b-8192',
        messages=[{
            "role": "system",
            "content": "Generate a comprehensive blog post from the following detailed bullet points."
        }, {
            "role": "user",
            "content": combined_summary
        }],
        temperature=0.85,
        max_tokens=3000,
        top_p=1,
        stream=True,
        stop=None
    )
    blog_post = "".join([chunk.choices[0].delta.content for chunk in completion if chunk.choices[0].delta.content is not None])
    return blog_post

# Streamlit app
def main():
    st.title("YouTube Transcript to Blog Post Generator")
    api_key = st.text_input("Enter your Groq API key:")
    video_url = st.text_input("Enter the URL of the YouTube video you want to create a blog post for:")
    if st.button("Generate Blog Post"):
        transcript_text = get_transcript_text(video_url)
        if transcript_text != "Invalid YouTube video URL":
            chunks = chunk_transcript(transcript_text)
            summaries = summarize_chunks(chunks, api_key)
            blog_post = generate_blog_post(summaries, api_key)
            st.markdown(blog_post)

if __name__ == "__main__":
    main()