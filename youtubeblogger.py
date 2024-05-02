import streamlit as st
from typing_extensions import Text
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from groq import Groq

def get_transcript_text(video_url):
    parsed_url = urlparse(video_url)
    video_id = parse_qs(parsed_url.query).get("v", [None])[0]

    if video_id:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join(line["text"] for line in transcript)
        return transcript_text
    else:
        return "Invalid YouTube video URL"

def generate_blog_post(transcript_text, api_key):
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are a blog post creator. \nYou will receive a raw transcript of a speech or video with information. Your job is to create a digestible, bulleted blog post with the information, without leaving anything out. "
            },
            {
                "role": "user",
                "content": "Create a blog post for the information below: "
            },
            {
                "role": "user",
                "content": transcript_text
            }
        ],
        temperature=0.85,
        max_tokens=1350,
        top_p=1,
        stream=True,
        stop=None,
    )

    blog_post = ""
    for chunk in completion:
        blog_post += chunk.choices[0].delta.content or ""

    return blog_post

# Streamlit app
def main():
    st.title("YouTube Transcript to Blog Post")

    # Get Groq API key from user
    api_key = st.text_input("Enter your Groq API key:")

    # Get YouTube video URL from user
    video_url = st.text_input("Enter the URL of the YouTube video you want to create a blog post for:")

    if st.button("Generate Blog Post"):
        # Get transcript text
        transcript_text = get_transcript_text(video_url)

        if transcript_text != "Invalid YouTube video URL":
            # Generate blog post
            blog_post = generate_blog_post(transcript_text, api_key)

            # Display blog post in Markdown
            st.markdown(blog_post)

if __name__ == "__main__":
    main()