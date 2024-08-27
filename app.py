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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image as PdfImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_pdf_file():
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name='TitleStyle', fontSize=24, alignment=1, spaceAfter=12)
    header_style = ParagraphStyle(name='HeaderStyle', fontSize=18, alignment=1, spaceAfter=12)
    normal_style = styles['Normal']
    normal_style.fontSize = 12

    # Add title
    elements.append(Paragraph("WhatsApp Chat Analysis Report", title_style))

    # Add images and charts
    image_elements = [
        ("Monthly Timeline", fig_monthly),
        ("Daily Timeline", fig_daily),
        ("Most Busy Day", fig_busy_day),
        ("Most Busy Month", fig_busy_month),
        ("Weekly Activity Map", fig_heatmap),
        ("Most Busy Users", fig_most_busy_users),
        ("Emoji Pie Chart", fig_emoji),
        ("Wordcloud", fig_wordcloud),
        ("Most Common Words", fig_most_common_words),
        ("Most Positive Users", fig_most_positive_users),
        ("Most Neutral Users", fig_most_neutral_users),
        ("Most Negative Users", fig_most_negative_users),
    ]

    for title, fig in image_elements:
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        img = PdfImage(img_buffer, width=6*inch, height=4*inch)
        elements.append(Paragraph(title, header_style))
        elements.append(img)
        elements.append(Paragraph("\n", normal_style))  # Add space between images

    # Add dataframes as tables
    dataframes = {
        "Emoji Analysis": emoji_df,
        "Most Busy Users": new_df,
        "Most Common Words": most_common_df
    }

    for title, df in dataframes.items():
        elements.append(Paragraph(title, header_style))
        table_data = [df.columns.tolist()] + df.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
        elements.append(Paragraph("\n", normal_style))  # Add space between tables

    pdf.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# Set page configuration
st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")

st.sidebar.title("WhatsApp Chat Analyzer")
# Load and display logo
logo_path = r"images/whatsapp-logo.png"  # Replace with the correct path and extension

# Open the image file
logo = Image.open(logo_path)

# Replace with the path to your logo file
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
                st.markdown("<div class='metric-card'><h4>Total Mssgs</h4><h2>{}</h2></div>".format(num_messages), unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='metric-card'><h4>Total Words</h4><h2>{}</h2></div>".format(words), unsafe_allow_html=True)
            with col3:
                st.markdown("<div class='metric-card'><h4>Media Shared</h4><h2>{}</h2></div>".format(num_media_messages), unsafe_allow_html=True)
            with col4:
                st.markdown("<div class='metric-card'><h4>Links Shared</h4><h2>{}</h2></div>".format(num_links), unsafe_allow_html=True)

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
            pdf_data = create_pdf_file()
            st.download_button(
                label="Download All Outputs as PDF",
                data=pdf_data,
                file_name="analysis_outputs.pdf",
                mime="application/pdf"
            )
