"""
app.py
======
Production-Ready WhatsApp Chat Analyzer Dashboard
Built with Streamlit, Plotly, Pandas, and NLTK.
"""

import os
import streamlit as st
import pandas as pd
import preprocessor
import helper

# -----------------------------------------------------------------------------
# Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern metrics and dashboard cards
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(37, 211, 102, 0.08) 0%, rgba(18, 140, 126, 0.15) 100%);
        border: 1px solid rgba(37, 211, 102, 0.25);
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #888888;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #25D366;
    }
    .date-badge {
        display: inline-block;
        background: rgba(37, 211, 102, 0.15);
        color: #25D366;
        border: 1px solid rgba(37, 211, 102, 0.3);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .quote-box {
        border-left: 4px solid #25D366;
        background-color: rgba(37, 211, 102, 0.06);
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
        font-style: italic;
    }
    .quote-author {
        font-weight: 700;
        font-style: normal;
        color: #128C7E;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar Configuration & Instructions
# -----------------------------------------------------------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg",
    width=64
)
st.sidebar.title("Chat Analyzer")
st.sidebar.caption("Production NLP & Statistical Intelligence Engine")

st.sidebar.markdown("---")

# Instructions Guide
with st.sidebar.expander("📖 How to Export Chat from WhatsApp", expanded=False):
    st.markdown("""
    **📱 Android:**
    1. Open any WhatsApp chat or group.
    2. Tap the **three dots (⋮)** in the top right corner.
    3. Select **More** → **Export chat**.
    4. Choose **Without Media** *(crucial for analysis)*.
    5. Save or send the generated `.txt` file.

    **🍎 iOS (iPhone):**
    1. Open the WhatsApp chat or group.
    2. Tap the contact or group name at the top.
    3. Scroll down and tap **Export Chat**.
    4. Select **Without Media**.
    5. Save the `.txt` file to Files or share via AirDrop/Email.
    """)

# File Uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload WhatsApp Chat (.txt)",
    type=["txt"],
    help="Upload an exported WhatsApp chat file in .txt format (without media)."
)

# Demo File Option
use_demo = st.sidebar.checkbox("📂 Or load sample project chat", value=False)

raw_data = None
sample_path = os.path.join(os.path.dirname(__file__), "sample_chat.txt")

if uploaded_file is not None:
    try:
        raw_data = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        # Fallback for Windows ANSI / Latin-1 exports
        raw_data = uploaded_file.getvalue().decode("latin-1")
elif use_demo and os.path.exists(sample_path):
    with open(sample_path, "r", encoding="utf-8") as f:
        raw_data = f.read()
    st.sidebar.info("💡 Loaded built-in sample demo chat.")

# -----------------------------------------------------------------------------
# Main Dashboard Application Logic
# -----------------------------------------------------------------------------
if raw_data is not None:
    try:
        with st.spinner("Parsing timestamps and processing message structures..."):
            df = preprocessor.preprocess(raw_data)
    except Exception as e:
        st.error(f"❌ Failed to parse chat file: {str(e)}")
        st.stop()

    if df.empty:
        st.warning("⚠️ No valid messages could be parsed from the file.")
        st.stop()

    # User List for Sidebar Selector
    user_list = sorted([u for u in df['user'].unique() if u != 'System'])
    user_list.insert(0, "Overall")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Analysis Scope")
    selected_user = st.sidebar.selectbox("Analyze for:", user_list)

    is_group_chat = len(user_list) > 2

    # Date Range metadata
    start_date, end_date, span_days = helper.fetch_active_date_range(df)

    # -------------------------------------------------------------------------
    # Dashboard Header & Metadata
    # -------------------------------------------------------------------------
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.title("📊 WhatsApp Chat Intelligence Dashboard")
        st.markdown(
            f"<div class='date-badge'>🗓️ Active Period: <b>{start_date}</b> — <b>{end_date}</b> ({span_days} days)</div>",
            unsafe_allow_html=True
        )
    with header_col2:
        st.metric(
            label="Selected View",
            value="Group Overall" if selected_user == "Overall" else selected_user
        )

    # Top KPI Metrics Cards
    stats = helper.fetch_stats(selected_user, df)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💬 Total Messages</div>
            <div class="metric-value">{stats['total_messages']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📝 Total Words</div>
            <div class="metric-value">{stats['total_words']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📁 Media Shared</div>
            <div class="metric-value">{stats['total_media']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🔗 Links Shared</div>
            <div class="metric-value">{stats['total_links']:,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # Navigation Tabs
    # -------------------------------------------------------------------------
    tab_titles = ["📈 Activity & Timelines", "🔤 Words & Emojis", "🧠 Sentiment Analysis"]
    if selected_user == "Overall" and is_group_chat:
        tab_titles.insert(1, "👥 Member Engagement")
    tab_titles.append("🔍 Data Explorer")

    tabs = st.tabs(tab_titles)
    tab_idx = 0

    # -------------------------------------------------------------------------
    # TAB 1: Activity & Timelines
    # -------------------------------------------------------------------------
    with tabs[tab_idx]:
        tab_idx += 1
        st.subheader("Temporal Trends & Activity Patterns")

        # Monthly & Daily Timelines
        t_col1, t_col2 = st.columns(2)

        with t_col1:
            st.markdown("##### 📅 Monthly Activity Timeline")
            monthly_df = helper.monthly_timeline(selected_user, df)
            if not monthly_df.empty:
                fig_monthly = helper.build_timeline_chart(
                    monthly_df,
                    x_col='time',
                    y_col='message_count',
                    title=f"Monthly Trend ({selected_user})"
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
            else:
                st.info("Insufficient data for monthly timeline.")

        with t_col2:
            st.markdown("##### 📈 Daily Activity Timeline")
            daily_df = helper.daily_timeline(selected_user, df)
            if not daily_df.empty:
                fig_daily = helper.build_timeline_chart(
                    daily_df,
                    x_col='only_date',
                    y_col='message_count',
                    title=f"Daily Trend ({selected_user})"
                )
                st.plotly_chart(fig_daily, use_container_width=True)
            else:
                st.info("Insufficient data for daily timeline.")

        st.markdown("---")

        # Activity Distribution: Days of Week & Months of Year
        act_col1, act_col2 = st.columns(2)

        with act_col1:
            st.markdown("##### 🗓️ Busiest Days of the Week")
            busy_day_series = helper.week_activity_map(selected_user, df)
            busy_day_df = busy_day_series.reset_index()
            busy_day_df.columns = ['Day', 'Messages']
            fig_day = helper.build_bar_chart(
                busy_day_df,
                x_col='Day',
                y_col='Messages',
                title="Activity by Day of Week",
                color='#128C7E'
            )
            st.plotly_chart(fig_day, use_container_width=True)

        with act_col2:
            st.markdown("##### 📆 Busiest Months of the Year")
            busy_month_series = helper.month_activity_map(selected_user, df)
            busy_month_df = busy_month_series.reset_index()
            busy_month_df.columns = ['Month', 'Messages']
            fig_month = helper.build_bar_chart(
                busy_month_df,
                x_col='Month',
                y_col='Messages',
                title="Activity by Month",
                color='#25D366'
            )
            st.plotly_chart(fig_month, use_container_width=True)

        st.markdown("---")

        # Weekly Hourly Heatmap
        st.markdown("##### 🔥 Weekly Hourly Activity Heatmap")
        heatmap_data = helper.activity_heatmap(selected_user, df)
        fig_heat = helper.build_heatmap_chart(
            heatmap_data,
            title=f"Activity Heatmap (Days vs Hours) — {selected_user}"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB: Member Engagement (Group Level only)
    # -------------------------------------------------------------------------
    if selected_user == "Overall" and is_group_chat:
        with tabs[tab_idx]:
            tab_idx += 1
            st.subheader("👥 Group Member Participation Rankings")

            counts_series, pct_df = helper.most_busy_users(df, top_n=10)

            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown("##### 🏆 Top Most Active Contributors")
                counts_df = counts_series.reset_index()
                counts_df.columns = ['User', 'Messages']
                fig_users = helper.build_bar_chart(
                    counts_df,
                    x_col='Messages',
                    y_col='User',
                    title="Message Count by User",
                    color='#25D366',
                    horizontal=True
                )
                st.plotly_chart(fig_users, use_container_width=True)

            with c2:
                st.markdown("##### 📋 Contribution Breakdown")
                st.dataframe(
                    pct_df,
                    use_container_width=True,
                    hide_index=True
                )

    # -------------------------------------------------------------------------
    # TAB: Words & Emojis
    # -------------------------------------------------------------------------
    with tabs[tab_idx]:
        tab_idx += 1
        st.subheader("Textual Linguistics & Emoji Analytics")

        # Word Cloud
        st.markdown("##### ☁️ High-Frequency Word Cloud")
        with st.spinner("Generating Word Cloud..."):
            fig_wc = helper.create_wordcloud(selected_user, df)
            st.pyplot(fig_wc)

        st.markdown("---")

        # Most Common Words
        st.markdown("##### 🔤 Top 20 Most Frequent Words")
        common_words_df = helper.most_common_words(selected_user, df, n=20)
        if not common_words_df.empty:
            fig_words = helper.build_bar_chart(
                common_words_df,
                x_col='Frequency',
                y_col='Word',
                title=f"Most Common Words ({selected_user})",
                color='#34B7F1',
                horizontal=True
            )
            st.plotly_chart(fig_words, use_container_width=True)
        else:
            st.info("No significant words found after stop-word filtering.")

        st.markdown("---")

        # Emoji Analytics
        st.markdown("##### 😀 Emoji Analytics")
        emoji_df = helper.emoji_helper(selected_user, df)

        if not emoji_df.empty:
            e_col1, e_col2 = st.columns([2, 3])
            with e_col1:
                total_emojis = emoji_df['Count'].sum()
                st.metric("Total Emojis Detected", f"{total_emojis:,}")
                st.metric("Unique Emojis Used", f"{len(emoji_df):,}")

                # Top 5 Emoji Pie Chart
                top5_emoji = emoji_df.head(5)
                fig_emoji_pie = helper.build_donut_chart(
                    labels=[f"{r['Emoji']} ({r['Description']})" for _, r in top5_emoji.iterrows()],
                    values=top5_emoji['Count'].tolist(),
                    title="Top 5 Emojis Distribution"
                )
                st.plotly_chart(fig_emoji_pie, use_container_width=True)

            with e_col2:
                st.markdown("###### Detailed Emoji Table")
                st.dataframe(
                    emoji_df,
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No emojis detected in this conversation scope.")

    # -------------------------------------------------------------------------
    # TAB: Sentiment Analysis (VADER)
    # -------------------------------------------------------------------------
    with tabs[tab_idx]:
        tab_idx += 1
        st.subheader("🧠 Sentiment & Emotional Tone Intelligence")

        with st.spinner("Analyzing sentiment with NLTK VADER..."):
            sentiment_res = helper.analyze_sentiment(selected_user, df)

        s_counts = sentiment_res['sentiment_counts']
        s_pct = sentiment_res['sentiment_pct']
        avg_comp = sentiment_res['avg_compound']

        # Metric Badges
        sent_col1, sent_col2, sent_col3, sent_col4 = st.columns(4)
        with sent_col1:
            st.metric("😊 Positive Messages", f"{s_counts['Positive']:,}", f"{s_pct['Positive']}%")
        with sent_col2:
            st.metric("😐 Neutral Messages", f"{s_counts['Neutral']:,}", f"{s_pct['Neutral']}%")
        with sent_col3:
            st.metric("😡 Negative Messages", f"{s_counts['Negative']:,}", f"{s_pct['Negative']}%")
        with sent_col4:
            st.metric("📊 Mean Sentiment Score", f"{avg_comp:+.3f}", help="VADER compound score [-1.0 to +1.0]")

        st.markdown("---")

        # Charts: Donut Distribution & User Sentiment Breakdown
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### 🍩 Overall Sentiment Distribution")
            fig_sent_donut = helper.build_donut_chart(
                labels=['Positive 😊', 'Neutral 😐', 'Negative 😡'],
                values=[s_counts['Positive'], s_counts['Neutral'], s_counts['Negative']],
                title=f"Sentiment Split ({selected_user})",
                colors=['#25D366', '#34B7F1', '#FF4B4B']
            )
            st.plotly_chart(fig_sent_donut, use_container_width=True)

        with chart_col2:
            if selected_user == "Overall" and not sentiment_res['user_sentiment_df'].empty:
                st.markdown("##### 👥 User-Wise Sentiment Scores")
                st.dataframe(
                    sentiment_res['user_sentiment_df'][['user', 'Positive', 'Neutral', 'Negative', 'Pos_Pct', 'Neg_Pct']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.markdown("##### 💡 Tone Summary")
                st.write(
                    f"Messages evaluated for **{selected_user}** show an average compound mood score of "
                    f"**`{avg_comp:+.3f}`**, indicating a predominantly "
                    f"**{'positive' if avg_comp > 0.05 else 'negative' if avg_comp < -0.05 else 'neutral'}** conversational mood."
                )

        st.markdown("---")

        # Top Positive & Top Negative Quotes
        q_col1, q_col2 = st.columns(2)
        with q_col1:
            st.markdown("##### 🌟 Most Positive Messages")
            if sentiment_res['top_positive']:
                for item in sentiment_res['top_positive']:
                    st.markdown(f"""
                    <div class="quote-box">
                        <div class="quote-author">👤 {item['user']} (Score: +{item['score']})</div>
                        "{item['message']}"
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No strongly positive messages found.")

        with q_col2:
            st.markdown("##### ⚠️ Most Critical / Negative Messages")
            if sentiment_res['top_negative']:
                for item in sentiment_res['top_negative']:
                    st.markdown(f"""
                    <div class="quote-box" style="border-left-color: #FF4B4B; background-color: rgba(255, 75, 75, 0.06);">
                        <div class="quote-author" style="color: #FF4B4B;">👤 {item['user']} (Score: {item['score']})</div>
                        "{item['message']}"
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No strongly negative messages found.")

    # -------------------------------------------------------------------------
    # TAB: Data Explorer
    # -------------------------------------------------------------------------
    with tabs[tab_idx]:
        st.subheader("🔍 Cleaned Chat Dataset Explorer")

        view_df = df if selected_user == "Overall" else df[df['user'] == selected_user]

        # Quick Search / Filter
        search_query = st.text_input("🔎 Search in messages:", "")
        if search_query:
            view_df = view_df[view_df['message'].str.contains(search_query, case=False, na=False)]

        st.dataframe(
            view_df[['date', 'user', 'message', 'is_media', 'is_deleted']],
            use_container_width=True,
            hide_index=True
        )

        # CSV Download Button
        csv_bytes = view_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_bytes,
            file_name=f"whatsapp_analysis_{selected_user.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

else:
    # Landing Screen when no file is uploaded yet
    st.info("👋 Welcome! Upload an exported WhatsApp `.txt` chat file in the left sidebar or check **'Load sample project chat'** to view the live dashboard.")

    l_col1, l_col2, l_col3 = st.columns(3)
    with l_col1:
        st.markdown("### 📊 Comprehensive Metrics")
        st.write("Track total messages, vocabulary volume, media shared, URLs, and active date spans across group and 1-on-1 chats.")
    with l_col2:
        st.markdown("### 📈 Interactive Plotly Charts")
        st.write("Explore daily and monthly timeline evolutions, day-of-week volume, and granular 24-hour hourly activity heatmaps.")
    with l_col3:
        st.markdown("### 🧠 NLP & VADER Sentiment")
        st.write("Extract emojis, generate stopword-filtered word clouds, and compute sentiment scores and tone distributions.")
