import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import io
import sys
import json
import requests
import datetime
import streamlit as st

# --- Initial Setup for NLTK ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
except Exception as e:
    st.error(f"An error occurred with NLTK download: {e}")
    st.stop()

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
except Exception as e:
    st.error(f"An error occurred with NLTK download: {e}")
    st.stop()


# --- Functions for Analysis ---
def parse_chat_file(file_content):
    data = {'date': [], 'time': [], 'ampm': [], 'user': [], 'message': []}
    
    pattern = re.compile(r'(\d{2}/\d{2}/\d{2}), (\d{1,2}:\d{2})\s(am|pm)\s-\s(.+?):\s(.+)')

    for line in file_content.split('\n'):
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if match:
            date, time, ampm, user, message = match.groups()
            data['date'].append(date)
            data['time'].append(time)
            data['ampm'].append(ampm)
            data['user'].append(user)
            data['message'].append(message)
        else:
            if data['message']:
                data['message'][-1] += ' ' + line

    df = pd.DataFrame(data)
    
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'] + ' ' + df['ampm'], format='%d/%m/%y %I:%M %p')
    
    return df

def generate_ai_summary(data_dict):
    system_prompt = "You are a world-class AI specialized in social media and chat analysis. Your task is to provide a concise, human-readable summary of a chat conversation based on the provided data. The tone should be friendly and conversational. Focus on the relationship dynamics, communication style, and emotional tone. Do not use technical jargon like 'compound scores'. Explain the data in a way a layperson can understand. The relationship should be described in human terms (e.g., 'harmonious', 'tense', 'functional')."
    user_query = f"""
    Please analyze the following chat data and provide a summary:
    - Total messages: {data_dict['total_messages']}
    - Number of users: {data_dict['num_users']}
    - Message counts per user: {data_dict['messages_per_user_markdown']}
    - Average message length: {data_dict['avg_word_count_markdown']}
    - Sentiment counts: {data_dict['sentiment_counts_markdown']}
    - Average sentiment score per user: {data_dict['sentiment_by_user_markdown']}
    - Most common emojis: {data_dict['emoji_counts']}
    - Most active day: {data_dict['most_active_day']}
    - Most active time of day: {data_dict['most_active_time']}
    - Top 5 most active days: {data_dict['key_events_markdown']}
    - Average time between messages: {data_dict['avg_reply_time_markdown']}
    - Longest active streak: {data_dict['active_streaks_markdown']}
    """
    
    apiKey = st.secrets["API_KEY"]
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"google_search": {}}]
    }

    try:
        response = requests.post(apiUrl, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status()
        result = response.json()
        candidate = result.get('candidates', [{}])[0]
        
        if candidate and candidate.get('content', {}).get('parts'):
            return candidate['content']['parts'][0]['text']
        else:
            return "AI summary could not be generated."
    except requests.exceptions.RequestException as e:
        return f"An error occurred during the API call: {e}"

def find_active_streaks(user_df):
    user_df = user_df.sort_values('datetime')
    user_df['time_gap'] = user_df['datetime'].diff().dt.total_seconds()
    user_df['streak_group'] = (user_df['time_gap'] > 60 * 60 * 24).cumsum()
    streaks = user_df.groupby('streak_group').size()
    if not streaks.empty:
        return streaks.max()
    return 0


# --- Streamlit App Layout ---
st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.title("WhatsApp Chat Analyzer")
st.markdown("Upload your chat file to get a full AI-powered analysis of your conversation.")


uploaded_file = st.file_uploader("Choose a chat.txt file", type="txt")

if uploaded_file is not None:
    with st.spinner('Analyzing chat... This might take a moment.'):
        # Read file content
        string_data = uploaded_file.getvalue().decode("utf-8")
        
        # Data Processing
        df = parse_chat_file(string_data)

        # Basic Stats
        messages_per_user = df['user'].value_counts()
        df['word_count'] = df['message'].apply(lambda s: len(s.split()))
        words_per_user = df.groupby('user')['word_count'].sum()
        media_messages = df[df['message'] == '<Media omitted>'].shape[0]
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        df['links'] = df['message'].apply(lambda x: len(re.findall(url_pattern, x)))
        total_links = df['links'].sum()
        avg_word_count = df.groupby('user')['word_count'].mean().sort_values(ascending=False)
        all_emojis = df['message'].apply(lambda x: [c for c in str(x) if c in emoji.EMOJI_DATA]).explode()
        emoji_counts = Counter(all_emojis.dropna())
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['hour'] = df['datetime'].dt.hour
        df['date_only'] = df['datetime'].dt.date
        df['month_only'] = df['datetime'].dt.strftime('%Y-%m')
        daily_activity = df.groupby('date_only').size()
        monthly_activity = df.groupby('month_only').size()
        key_events = daily_activity.nlargest(5).reset_index()
        df['time_of_day'] = df['hour'].apply(lambda hour: 'Morning' if 5 <= hour < 12 else 'Afternoon' if 12 <= hour < 17 else 'Evening' if 17 <= hour < 21 else 'Night')
        peak_hours = df['time_of_day'].value_counts()
        activity_heatmap_data = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        analyzer = SentimentIntensityAnalyzer()
        df['sentiment_score'] = df['message'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
        def get_sentiment_label(score):
            return 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral'
        df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)
        sentiment_by_user = df.groupby('user')['sentiment_score'].mean().sort_values(ascending=False)
        df['time_diff'] = df['datetime'].diff().dt.total_seconds() / 60
        avg_reply_time = df.groupby('user')['time_diff'].mean().sort_values()
        active_streaks = df.groupby('user').apply(find_active_streaks)
        
        # AI Summary
        data_dict = {
            'total_messages': df.shape[0],
            'num_users': len(messages_per_user),
            'messages_per_user_markdown': messages_per_user.to_markdown(),
            'avg_word_count_markdown': avg_word_count.to_markdown(),
            'sentiment_counts_markdown': df['sentiment_label'].value_counts().to_markdown(),
            'sentiment_by_user_markdown': sentiment_by_user.to_markdown(),
            'emoji_counts': emoji_counts.most_common(5),
            'most_active_day': df['day_of_week'].mode()[0] if not df['day_of_week'].mode().empty else 'N/A',
            'most_active_time': df['time_of_day'].mode()[0] if not df['time_of_day'].mode().empty else 'N/A',
            'key_events_markdown': key_events.to_markdown(index=False),
            'avg_reply_time_markdown': avg_reply_time.to_markdown(),
            'active_streaks_markdown': active_streaks.to_markdown(),
        }

        ai_summary = generate_ai_summary(data_dict)

        # --- Display Output ---
        st.success("Analysis Complete!")
        st.write("---")

        st.header("AI-Powered Conversation Summary")
        st.write(ai_summary)
        
        st.write("---")

        st.header("Key Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", df.shape[0])
        col2.metric("Total Words", df['word_count'].sum())
        col3.metric("Media Shared", media_messages)
        col4.metric("Links Shared", total_links)

        st.write("---")
        st.header("User Insights")
        st.subheader("Messages per user")
        st.dataframe(messages_per_user.to_frame(), use_container_width=True)

        st.subheader("Average Message Length (in words)")
        st.dataframe(avg_word_count.to_frame(), use_container_width=True)

        st.subheader("Average Time Between Messages")
        st.dataframe(avg_reply_time.to_frame(), use_container_width=True)

        st.subheader("Active Streaks")
        st.dataframe(active_streaks.to_frame(), use_container_width=True)

        st.subheader("Key Events")
        st.dataframe(key_events, use_container_width=True)

        st.subheader("Peak Hours")
        st.dataframe(peak_hours.to_frame(), use_container_width=True)
        
        st.subheader("Top 10 Emojis Used")
        emoji_df = pd.DataFrame(emoji_counts.most_common(10), columns=['Emoji', 'Count'])
        st.dataframe(emoji_df, use_container_width=True)

        st.write("---")
        st.header("Chat Activity & Trends")
        
        fig, axs = plt.subplots(2, 1, figsize=(18, 16)) 

        # --- Plot 1: Monthly Message Trends ---
        monthly_activity = df.groupby(df['datetime'].dt.to_period('M')).size()
        monthly_activity.index = monthly_activity.index.astype(str)
        sns.barplot(x=monthly_activity.index, y=monthly_activity.values, color='deepskyblue', ax=axs[0])
        axs[0].set_title('Monthly Message Trends', fontsize=20, fontweight='bold')
        axs[0].set_xlabel('Month', fontsize=14)
        axs[0].set_ylabel('Message Count', fontsize=14)
        axs[0].tick_params(axis='x', rotation=45, labelsize=12)
        axs[0].tick_params(axis='y', labelsize=12)
        axs[0].grid(True, linestyle='--', alpha=0.7, axis='y')
        axs[0].margins(x=0.02) 

        # --- Plot 2: Most Active Day and Time Heatmap ---
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        activity_heatmap_data_ordered = activity_heatmap_data.reindex(day_order)

        sns.heatmap(activity_heatmap_data_ordered, cmap='YlGnBu', annot=True, fmt='d', ax=axs[1],
                    linewidths=.5, linecolor='black', cbar_kws={'shrink': .8}) 
        axs[1].set_title('Most Active Day and Time Heatmap (Message Frequency)', fontsize=20, fontweight='bold')
        axs[1].set_xlabel('Hour of Day (0-23)', fontsize=14)
        axs[1].set_ylabel('Day of Week', fontsize=14)
        axs[1].tick_params(axis='x', labelsize=12)
        axs[1].tick_params(axis='y', labelsize=12, rotation=0)
        
        plt.tight_layout(pad=4.0)

        st.pyplot(fig)
        
    st.write("---")
