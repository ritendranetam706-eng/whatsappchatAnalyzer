"""
helper.py
=========
Statistical analytics, NLP pipelines, and interactive Plotly chart builders
for the WhatsApp Chat Analyzer.
"""

from collections import Counter
import re
from typing import Dict, Tuple, Any

import emoji
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# NLTK and VADER Sentiment
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure required NLTK resources are loaded safely
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

try:
    from nltk.corpus import stopwords
    nltk_stopwords = set(stopwords.words('english'))
except (LookupError, AttributeError):
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    nltk_stopwords = set(stopwords.words('english'))

# Initialize VADER analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Extended stopwords including common chat colloquialisms and Hinglish terms
CUSTOM_STOPWORDS = {
    'media', 'omitted', 'message', 'deleted', 'image', 'video', 'audio',
    'document', 'sticker', 'attached', 'call', 'missed', 'voice',
    'ok', 'okay', 'yes', 'no', 'yeah', 'yep', 'nope', 'lol', 'lmao',
    'rofl', 'haha', 'hahaha', 'k', 'kk', 'hmm', 'hmmm', 'hi', 'hello',
    'hey', 'good', 'morning', 'night', 'thanks', 'thank', 'welcome',
    'please', 'bro', 'dude', 'guy', 'guys', 'bhai', 'yaar', 'hai', 'hain',
    'bhi', 'aur', 'kya', 'ko', 'se', 'ne', 'ka', 'ke', 'ki', 'ye', 'yeh',
    'wo', 'woh', 'tha', 'thi', 'the', 'hoga', 'hogi', 'kar', 'karo',
    'raha', 'rahi', 'kuch', 'ab', 'toh', 'na', 'nahi', 'nhi', 'mera',
    'meri', 'mere', 'apna', 'apni', 'tere', 'teri', 'tera', 'pe', 'par',
    'unka', 'unki', 'sab', 'sirf', 'ek', 'do', 'teen'
}

TOTAL_STOPWORDS = nltk_stopwords.union(CUSTOM_STOPWORDS)

# URL matching regex
URL_REGEX = re.compile(
    r'(https?://(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'
    r'www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'
    r'https?://[^\s]+|'
    r'ftp://[^\s]+)',
    re.IGNORECASE
)

# Standard days and months for ordered indexing
DAYS_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
MONTHS_ORDER = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']


# -----------------------------------------------------------------------------
# 1. Top Statistics & Metadata
# -----------------------------------------------------------------------------

def fetch_stats(selected_user: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary metrics: total messages, total words, media count, links count.

    Parameters:
        selected_user (str): 'Overall' or a specific username.
        df (pd.DataFrame): Preprocessed chat DataFrame.

    Returns:
        dict: Summary metrics.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Filter out System messages for user-level stats
    active_df = df[df['user'] != 'System'] if selected_user == 'Overall' else df

    total_messages = active_df.shape[0]

    # Count words only in non-media, non-deleted messages
    text_messages = active_df[~active_df['is_media'] & ~active_df['is_deleted']]['message']
    total_words = sum(len(msg.split()) for msg in text_messages)

    # Total media shared
    total_media = active_df['is_media'].sum()

    # Total URLs extracted
    total_links = 0
    for msg in text_messages:
        links = URL_REGEX.findall(msg)
        total_links += len(links)

    return {
        'total_messages': int(total_messages),
        'total_words': int(total_words),
        'total_media': int(total_media),
        'total_links': int(total_links)
    }


def fetch_active_date_range(df: pd.DataFrame) -> Tuple[str, str, int]:
    """
    Returns start date, end date, and total active span in days.
    """
    if df.empty or 'date' not in df.columns:
        return ("N/A", "N/A", 0)

    min_date = df['date'].min()
    max_date = df['date'].max()
    span_days = (max_date - min_date).days + 1

    start_str = min_date.strftime("%d %b %Y")
    end_str = max_date.strftime("%d %b %Y")

    return (start_str, end_str, span_days)


# -----------------------------------------------------------------------------
# 2. Activity Timelines & Temporal Maps
# -----------------------------------------------------------------------------

def monthly_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates message volume by month and year.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['user'] != 'System']

    timeline = (
        df.groupby(['year', 'month_num', 'month'])
        .size()
        .reset_index(name='message_count')
    )

    # Format sortable chronological time column
    timeline['time'] = timeline['month'] + ' ' + timeline['year'].astype(str)
    timeline = timeline.sort_values(by=['year', 'month_num']).reset_index(drop=True)
    return timeline


def daily_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates message volume by exact date.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['user'] != 'System']

    daily = (
        df.groupby('only_date')
        .size()
        .reset_index(name='message_count')
    )
    daily['only_date'] = pd.to_datetime(daily['only_date'])
    daily = daily.sort_values('only_date').reset_index(drop=True)
    return daily


def week_activity_map(selected_user: str, df: pd.DataFrame) -> pd.Series:
    """
    Computes message count by day of the week (Monday through Sunday).
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['user'] != 'System']
    counts = df['day_name'].value_counts()
    return counts.reindex(DAYS_ORDER, fill_value=0)


def month_activity_map(selected_user: str, df: pd.DataFrame) -> pd.Series:
    """
    Computes message count by month of the year (January through December).
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['user'] != 'System']
    counts = df['month'].value_counts()
    return counts.reindex(MONTHS_ORDER, fill_value=0)


def activity_heatmap(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a pivot table of Day of Week vs. Hourly Periods (00-01 to 23-00).
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['user'] != 'System']

    all_periods = [f"{h:02d}-{(h + 1) % 24:02d}" for h in range(24)]

    heatmap_df = df.pivot_table(
        index='day_name',
        columns='period',
        values='message',
        aggfunc='count'
    ).fillna(0)

    # Reindex to ensure consistent shape and chronological order
    heatmap_df = heatmap_df.reindex(index=DAYS_ORDER, fill_value=0)
    for p in all_periods:
        if p not in heatmap_df.columns:
            heatmap_df[p] = 0
    heatmap_df = heatmap_df[all_periods]

    return heatmap_df


# -----------------------------------------------------------------------------
# 3. User Engagement (Group Level)
# -----------------------------------------------------------------------------

def most_busy_users(df: pd.DataFrame, top_n: int = 10) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calculates member activity rankings and contribution percentages.
    """
    non_sys = df[df['user'] != 'System']
    counts = non_sys['user'].value_counts().head(top_n)

    total_msgs = non_sys.shape[0]
    pct = ((counts / total_msgs) * 100).round(2).reset_index()
    pct.columns = ['User', 'Percentage (%)']

    count_df = counts.reset_index()
    count_df.columns = ['User', 'Message Count']

    result_df = pd.merge(count_df, pct, on='User')
    return counts, result_df


# -----------------------------------------------------------------------------
# 4. Textual NLP Analysis (Word Cloud & Frequencies)
# -----------------------------------------------------------------------------

def _clean_text_corpus(selected_user: str, df: pd.DataFrame) -> list:
    """
    Helper function to filter and clean message tokens.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Filter out system, media, deleted messages
    filtered_df = df[
        (df['user'] != 'System') &
        (~df['is_media']) &
        (~df['is_deleted'])
    ]

    words = []
    for msg in filtered_df['message']:
        # Remove URLs
        clean_msg = URL_REGEX.sub('', msg)
        # Tokenize by alphanumeric words
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', clean_msg.lower())
        for word in tokens:
            if word not in TOTAL_STOPWORDS:
                words.append(word)

    return words


def create_wordcloud(selected_user: str, df: pd.DataFrame) -> plt.Figure:
    """
    Generates a high-resolution Matplotlib WordCloud figure.
    """
    words = _clean_text_corpus(selected_user, df)
    text = " ".join(words)

    if not text.strip():
        text = "No words available for wordcloud"

    wc = WordCloud(
        width=1000,
        height=500,
        background_color='#0f172a',
        colormap='viridis',
        stopwords=TOTAL_STOPWORDS,
        min_font_size=10,
        max_words=200,
        prefer_horizontal=0.85,
        collocations=False
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0f172a')
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


def most_common_words(selected_user: str, df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """
    Returns the top N most frequent words excluding stop words.
    """
    words = _clean_text_corpus(selected_user, df)
    most_common = Counter(words).most_common(n)

    if not most_common:
        return pd.DataFrame(columns=['Word', 'Frequency'])

    return pd.DataFrame(most_common, columns=['Word', 'Frequency'])


def emoji_helper(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts, counts, and describes emojis used in chat messages.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['user'] != 'System']

    emojis_list = []
    for msg in df['message']:
        emojis_list.extend([c for c in msg if emoji.is_emoji(c)])

    emoji_counts = Counter(emojis_list).most_common()

    if not emoji_counts:
        return pd.DataFrame(columns=['Emoji', 'Count', 'Description'])

    emoji_df = pd.DataFrame(emoji_counts, columns=['Emoji', 'Count'])
    emoji_df['Description'] = emoji_df['Emoji'].apply(
        lambda e: emoji.demojize(e).replace(':', '').replace('_', ' ').capitalize()
    )
    return emoji_df


# -----------------------------------------------------------------------------
# 5. VADER Sentiment Analysis
# -----------------------------------------------------------------------------

def analyze_sentiment(selected_user: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs VADER sentiment classification on text messages.
    Categorizes messages into Positive (compound >= 0.05),
    Negative (compound <= -0.05), and Neutral.

    Returns:
        dict containing:
            - sentiment_counts: dict {Positive, Neutral, Negative}
            - sentiment_pct: dict with percentages
            - avg_compound: float
            - user_sentiment_df: pd.DataFrame (user-level sentiment scores)
            - top_positive: list of dicts (message, user, score)
            - top_negative: list of dicts (message, user, score)
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Only analyze text messages
    valid_msgs = df[
        (df['user'] != 'System') &
        (~df['is_media']) &
        (~df['is_deleted']) &
        (df['message'].str.strip().str.len() > 1)
    ].copy()

    if valid_msgs.empty:
        return {
            'sentiment_counts': {'Positive': 0, 'Neutral': 0, 'Negative': 0},
            'sentiment_pct': {'Positive': 0.0, 'Neutral': 0.0, 'Negative': 0.0},
            'avg_compound': 0.0,
            'user_sentiment_df': pd.DataFrame(),
            'top_positive': [],
            'top_negative': []
        }

    # Calculate VADER scores
    def get_vader_scores(text: str):
        # Strip URL before scoring
        clean_text = URL_REGEX.sub('', text).strip()
        return vader_analyzer.polarity_scores(clean_text)

    scores = valid_msgs['message'].apply(get_vader_scores)
    valid_msgs['compound'] = scores.apply(lambda s: s['compound'])
    valid_msgs['pos'] = scores.apply(lambda s: s['pos'])
    valid_msgs['neu'] = scores.apply(lambda s: s['neu'])
    valid_msgs['neg'] = scores.apply(lambda s: s['neg'])

    def categorize(score: float) -> str:
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        return 'Neutral'

    valid_msgs['sentiment'] = valid_msgs['compound'].apply(categorize)

    counts = valid_msgs['sentiment'].value_counts().to_dict()
    counts.setdefault('Positive', 0)
    counts.setdefault('Neutral', 0)
    counts.setdefault('Negative', 0)

    total_scored = len(valid_msgs)
    pct = {k: round((v / total_scored) * 100, 1) for k, v in counts.items()}
    avg_compound = round(float(valid_msgs['compound'].mean()), 3)

    # User-level sentiment breakdown
    user_sent = (
        valid_msgs.groupby(['user', 'sentiment'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ['Positive', 'Neutral', 'Negative']:
        if col not in user_sent.columns:
            user_sent[col] = 0

    user_sent['Total'] = user_sent['Positive'] + user_sent['Neutral'] + user_sent['Negative']
    user_sent['Pos_Pct'] = ((user_sent['Positive'] / user_sent['Total']) * 100).round(1)
    user_sent['Neg_Pct'] = ((user_sent['Negative'] / user_sent['Total']) * 100).round(1)
    user_sent['Neu_Pct'] = ((user_sent['Neutral'] / user_sent['Total']) * 100).round(1)
    user_sent = user_sent.sort_values(by='Total', ascending=False)

    # Top positive & negative quotes (minimum word length of 3 to avoid single word artifacts)
    multiword_msgs = valid_msgs[valid_msgs['message'].str.split().str.len() >= 3]

    top_pos_df = multiword_msgs.sort_values('compound', ascending=False).head(3)
    top_pos = [
        {'user': r['user'], 'message': r['message'], 'score': round(r['compound'], 3)}
        for _, r in top_pos_df.iterrows()
    ]

    top_neg_df = multiword_msgs.sort_values('compound', ascending=True).head(3)
    top_neg = [
        {'user': r['user'], 'message': r['message'], 'score': round(r['compound'], 3)}
        for _, r in top_neg_df.iterrows()
    ]

    return {
        'sentiment_counts': counts,
        'sentiment_pct': pct,
        'avg_compound': avg_compound,
        'user_sentiment_df': user_sent,
        'top_positive': top_pos,
        'top_negative': top_neg
    }


# -----------------------------------------------------------------------------
# 6. Interactive Plotly Chart Builders
# -----------------------------------------------------------------------------

def build_timeline_chart(df_timeline: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    """
    Creates a responsive Plotly line chart with gradient fill and custom hover tooltips.
    """
    fig = px.line(
        df_timeline,
        x=x_col,
        y=y_col,
        title=title,
        markers=True,
        color_discrete_sequence=['#25D366']
    )
    fig.update_traces(
        line=dict(width=2.5),
        marker=dict(size=6, color='#128C7E'),
        hovertemplate='<b>%{x}</b><br>Messages: %{y:,}<extra></extra>'
    )
    fig.update_layout(
        template='plotly_dark',
        hovermode='x unified',
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="",
        yaxis_title="Message Count",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)'
    )
    return fig


def build_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str,
                    color: str = '#25D366', horizontal: bool = False) -> go.Figure:
    """
    Creates a styled interactive bar chart.
    """
    if horizontal:
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            orientation='h',
            title=title,
            color_discrete_sequence=[color]
        )
        fig.update_traces(
            hovertemplate='<b>%{y}</b>: %{x:,}<extra></extra>'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title=x_col,
            yaxis_title=""
        )
    else:
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            title=title,
            color_discrete_sequence=[color]
        )
        fig.update_traces(
            hovertemplate='<b>%{x}</b>: %{y:,}<extra></extra>'
        )
        fig.update_layout(
            xaxis_title="",
            yaxis_title=y_col
        )

    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)'
    )
    return fig


def build_donut_chart(labels: list, values: list, title: str,
                      colors: list = None) -> go.Figure:
    """
    Creates an interactive donut / pie chart with custom slice colors and hover info.
    """
    if colors is None:
        colors = ['#25D366', '#34B7F1', '#FFA500', '#FF4B4B', '#9C27B0']

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(colors=colors),
                textinfo='percent+label',
                hoverinfo='label+value+percent'
            )
        ]
    )
    fig.update_layout(
        title=title,
        template='plotly_dark',
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    return fig


def build_heatmap_chart(heatmap_df: pd.DataFrame, title: str) -> go.Figure:
    """
    Builds an interactive 2D density heatmap of chat activity.
    """
    fig = px.imshow(
        heatmap_df,
        labels=dict(x="Hour of Day", y="Day of Week", color="Messages"),
        x=heatmap_df.columns,
        y=heatmap_df.index,
        title=title,
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Time Interval (24h)",
        yaxis_title=""
    )
    fig.update_traces(
        hovertemplate='<b>%{y}</b> @ %{x}<br>Messages: %{z}<extra></extra>'
    )
    return fig
