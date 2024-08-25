import streamlit as st
import preprocessor
import nltk
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from PIL import Image
import io
import time
import zipfile

def create_zip_file():
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, 'w') as zip_file:
        # Create and write images
        for analysis_name, fig in zip(
            ["Monthly_Timeline", "Daily_Timeline", "Most_Busy_Day", "Most_Busy_Month", "Weekly_Activity_Map","fig_most_busy_users","emoji_pie_chart", "Wordcloud", "Most_Common_Words","fig_most_positive_users","fig_most_neutral_users","fig_most_negative_users"],
            [fig_monthly, fig_daily, fig_busy_day, fig_busy_month, fig_heatmap, fig_most_busy_users,fig_emoji,fig_wordcloud, fig_most_common_words,fig_most_positive_users,fig_most_neutral_users,fig_most_negative_users]
        ):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png',bbox_inches='tight')
            img_buffer.seek(0)
            zip_file.writestr(f"{analysis_name}.png", img_buffer.read())

        # Create and write dataframes as CSV
        dataframes = {
            "Emoji_Analysis": emoji_df.to_csv(index=False),
            "Most_Busy_Users": new_df.to_csv(index=False),
            "Most_Common_Words": most_common_df.to_csv(index=False)
        }

        for name, csv in dataframes.items():
            zip_file.writestr(f"{name}.csv", csv)

    buffer.seek(0)
    return buffer.getvalue()

# Set page configuration
st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")

st.sidebar.title("WhatsApp Chat Analyzer")
# Load and display logo
logo_path = r"https://github.com/soumyaaich-04/Whatsapp_chat_analyzer_app/blob/main/whatsapp-logo.png"  # Replace with the correct path and extension

# Open the image file
logo = Image.open(logo_path)

# logo = Image.open("C:\Users\TANIYA\Downloads\wp pic")  # Replace with the path to your logo file
st.sidebar.image(logo, width=150)

# Custom CSS
st.markdown("""
    <style>

        .title {
            text-align: center;
            color: #25D366; /* WhatsApp green color */
            font-size: 36px;
            font-weight: bold;
            margin-top: 20px;
        }
        .header {
            font-size: 24px;
            color: #075E54; /* WhatsApp green color */
        }
        .metric-card {
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            color: black;
        }
        .subheader {
            font-size: 20px;
            color: #075E54; /* WhatsApp green color */
        }
        .stButton>button {
            background-color: #25D366; /* WhatsApp green color */
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #128C7E; /* Darker green */
        }
        .stMetric {
            font-size: 20px;
            font-weight: bold;
        }
        .stDataFrame {
            font-size: 16px;
        }
        /* Center align the logo in the sidebar */
        .css-1v3fvcr {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# Display header
st.markdown("<div class='title'>WhatsApp Chat Analyzer</div>", unsafe_allow_html=True)

# Download nltk resources
nltk.download('vader_lexicon')

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["txt"])

# Check if a file has been uploaded
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # Sentiment Analysis
    sentiments = SentimentIntensityAnalyzer()
    sentiment_data = {
        "po": [sentiments.polarity_scores(i)["pos"] for i in df['message']],
        "ne": [sentiments.polarity_scores(i)["neg"] for i in df['message']],
        "nu": [sentiments.polarity_scores(i)["neu"] for i in df['message']]
    }


    def sentiment(d):
        if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
            return 1
        if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
            return -1
        return 0


    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_df['value'] = sentiment_df.apply(lambda row: sentiment(row), axis=1)

    # Sidebar user selection
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    # Show analysis button
    if st.sidebar.button("Show Analysis"):
        with st.spinner('Processing...'):
            # Simulate a delay for demonstration purposes
            time.sleep(2)
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
            # Stats area
            st.title('Top Statistics')
            col1, col2, col3, col4 = st.columns(4)
            with col1:

                st.markdown("<div class='metric-card'><h4>Total Mssgs</h4><h2>{}</h2></div>".format(num_messages),
                            unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='metric-card'><h4>Total Words</h4><h2>{}</h2></div>".format(words),
                            unsafe_allow_html=True)
            with col3:
                st.markdown(
                    "<div class='metric-card'><h4>Media Shared</h4><h2>{}</h2></div>".format(num_media_messages),
                    unsafe_allow_html=True)
            with col4:
                st.markdown("<div class='metric-card'><h4>Links Shared</h4><h2>{}</h2></div>".format(num_links),
                            unsafe_allow_html=True)

            # Monthly Timeline
            st.title('Monthly Timeline')
            timeline = helper.monthly_timeline(selected_user, df)
            fig_monthly, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='#25D366')
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            st.pyplot(fig_monthly)

            # Daily Timeline
            st.title('Daily Timeline')
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig_daily, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='#128C7E')
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            st.pyplot(fig_daily)

            # Activity Map
            st.title('Activity Map')
            col1, col2 = st.columns(2)

            with col1:
                st.header('Most Busy Day')
                busy_day = helper.week_activity_map(selected_user, df)
                fig_busy_day, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values)
                plt.xticks(rotation='vertical')
                plt.tight_layout()
                st.pyplot(fig_busy_day)

            with col2:
                st.header('Most Busy Month')
                busy_month = helper.month_activity_map(selected_user, df)
                fig_busy_month, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='orange')
                plt.tight_layout()
                st.pyplot(fig_busy_month)

            st.title("Weekly Activity Map")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            fig_heatmap, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig_heatmap)

            # Finding the busiest users in the group
            if selected_user == 'Overall':
                st.title('Most Busy Users')
                x, new_df = helper.most_busy_users(df)
                fig_most_busy_users, ax = plt.subplots()
                col1, col2 = st.columns(2)

                with col1:
                    ax.bar(x.index, x.values)
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig_most_busy_users)
                with col2:
                    st.dataframe(new_df)

            # Sentiment Analysis
            st.title('Sentiment Analysis')
            if selected_user == 'Overall':
                x = df['user'][sentiment_df['value'] == 1].value_counts().head(10)
                y = df['user'][sentiment_df['value'] == -1].value_counts().head(10)
                z = df['user'][sentiment_df['value'] == 0].value_counts().head(10)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Most Positive Users")
                    fig_most_positive_users, ax = plt.subplots()
                    ax.bar(x.index, x.values, color='green')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig_most_positive_users)
                with col2:
                    st.subheader("Most Neutral Users")
                    fig_most_neutral_users, ax = plt.subplots()
                    ax.bar(z.index, z.values, color='grey')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig_most_neutral_users)
                with col3:
                    st.subheader("Most Negative Users")
                    fig_most_negative_users, ax = plt.subplots()
                    ax.bar(y.index, y.values, color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig_most_negative_users)

            # emoji analysis
            emoji_df = helper.emoji_helper(selected_user, df)
            st.title('Emoji Analysis')

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(emoji_df)
            with col2:
                def set_title_with_size(title_text, font_size="30px"):
                    html_title = f"""
                                <h1 style="font-size: {font_size};"> {title_text} </h1>
                            """
                    st.write(html_title, unsafe_allow_html=True)


                set_title_with_size("Top 5 most emojis used")
                fig_emoji, ax = plt.subplots()
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                st.pyplot(fig_emoji)

            # WordCloud
            st.title("Wordcloud")
            df_wc = helper.create_wordcloud(selected_user, df)
            fig_wordcloud, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig_wordcloud)

            # Most Common Words
            most_common_df = helper.most_common_words(selected_user, df)
            fig_most_common_words, ax = plt.subplots()
            ax.bar(most_common_df[0], most_common_df[1])
            plt.xticks(rotation='vertical')
            st.title('Most Common Words')
            st.pyplot(fig_most_common_words)

            # Create and provide download button
            zip_data = create_zip_file()
            st.download_button(
                label="Download All Outputs",
                data=zip_data,
                file_name="analysis_outputs.zip",
                mime="application/zip"
            )

