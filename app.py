import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv
import os

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="AI Support Operations",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# LOAD API KEY
# -----------------------------

load_dotenv()

api_key = st.secrets.get("OPENAI_API_KEY")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None


# -----------------------------
# LOAD DATA
# -----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/bizops_challenge_support_tickets.csv"
    )

    df["created_at"] = pd.to_datetime(df["created_at"])

    return df


df = load_data()


# -----------------------------
# HEADER
# -----------------------------

st.title("🤖 AI Support Operations")

st.write(
    "AI-powered support ticket triage, response generation, "
    "and trend detection."
)

st.divider()


# -----------------------------
# KPI DASHBOARD
# -----------------------------

total_tickets = len(df)

open_tickets = len(
    df[df["status"].isin(["open", "pending"])]
)

resolved_tickets = len(
    df[df["status"] == "resolved"]
)

avg_resolution = df["resolution_time_hours"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Tickets",
    total_tickets
)

col2.metric(
    "Open / Pending",
    open_tickets
)

col3.metric(
    "Resolved",
    resolved_tickets
)

col4.metric(
    "Avg. Resolution Time",
    f"{avg_resolution:.1f} hrs"
)


st.divider()


# -----------------------------
# TABS
# -----------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "Ticket Triage",
        "AI Response",
        "Trends & Root Causes"
    ]
)


# =====================================================
# TAB 1 — TICKET TRIAGE
# =====================================================

with tab1:

    st.header("AI Ticket Triage & Routing")

    st.write(
        "Enter a customer support ticket and the AI will "
        "classify the issue, determine urgency, and recommend "
        "the appropriate operations team."
    )

    ticket_text = st.text_area(
        "Customer ticket",
        placeholder=(
            "Example: I can't make a payment and my card "
            "keeps getting declined."
        ),
        height=150
    )

    if st.button(
        "Analyze Ticket",
        type="primary"
    ):

        if not ticket_text:

            st.warning(
                "Please enter a ticket first."
            )

        elif not client:

            st.error(
                "OpenAI API key not found. "
                "Check your .env file."
            )

        else:

            with st.spinner(
                "AI is analyzing the ticket..."
            ):

                prompt = f"""
You are an AI support operations assistant.

Analyze the following customer support ticket.

Ticket:
{ticket_text}

Return:

1. Product/category
2. Priority: Low, Medium, or High
3. Recommended operations team
4. Confidence from 0-100%
5. Short explanation

Possible products:
- Cards
- Payments
- Account
- FX

Use the ticket information to make the best judgment.
"""

                response = client.responses.create(
                    model="gpt-5.6-luna",
                    input=prompt
                )

                result = response.output_text

            st.subheader("AI Analysis")

            st.write(result)


# =====================================================
# TAB 2 — AI RESPONSE
# =====================================================

with tab2:

    st.header("AI Suggested Response")

    st.write(
        "Generate a response that a support agent can review "
        "before sending to the customer."
    )

    response_ticket = st.text_area(
        "Customer ticket",
        placeholder=(
            "Paste the customer's ticket here..."
        ),
        height=150
    )

    if st.button(
        "Generate Response",
        type="primary"
    ):

        if not response_ticket:

            st.warning(
                "Please enter a ticket first."
            )

        elif not client:

            st.error(
                "OpenAI API key not found."
            )

        else:

            with st.spinner(
                "Generating response..."
            ):

                prompt = f"""
You are a professional customer support assistant.

Write a concise, helpful response to this customer.

Customer ticket:
{response_ticket}

Requirements:
- Be professional and empathetic
- Do not invent policies or facts
- Do not promise something you cannot verify
- If additional information is required, ask for it
- Keep the response concise
"""

                response = client.responses.create(
                    model="gpt-5.6-luna",
                    input=prompt
                )

                answer = response.output_text

            st.subheader(
                "Suggested Response"
            )

            st.info(answer)

            st.caption(
                "Human review recommended before sending."
            )


# =====================================================
# TAB 3 — TRENDS
# =====================================================

with tab3:

    st.header("Ticket Trends & Root-Cause Detection")

    st.write(
        "Identify recurring issues, ticket concentrations, "
        "and potential emerging operational problems."
    )

    # Product distribution

    st.subheader("Tickets by Product")

    product_counts = (
        df["product"]
        .value_counts()
        .reset_index()
    )

    product_counts.columns = [
        "Product",
        "Tickets"
    ]

    fig = px.bar(
        product_counts,
        x="Product",
        y="Tickets",
        title="Support Tickets by Product"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Daily volume

    st.subheader("Ticket Volume Over Time")

    daily = (
        df.groupby(
            df["created_at"].dt.date
        )
        .size()
        .reset_index(
            name="Tickets"
        )
    )

    daily.columns = [
        "Date",
        "Tickets"
    ]

    fig2 = px.line(
        daily,
        x="Date",
        y="Tickets",
        markers=True,
        title="Daily Ticket Volume"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


    # Status

    st.subheader("Current Ticket Status")

    status_counts = (
        df["status"]
        .value_counts()
        .reset_index()
    )

    status_counts.columns = [
        "Status",
        "Tickets"
    ]

    st.dataframe(
        status_counts,
        use_container_width=True,
        hide_index=True
    )


    # Potential issues

    st.subheader(
        "🚨 Potential Emerging Issues"
    )

    recent = df[
        df["created_at"]
        >= df["created_at"].max()
        - pd.Timedelta(days=10)
    ]

    recent_products = (
        recent["product"]
        .value_counts()
    )

    for product, count in recent_products.items():

        st.write(
            f"**{product}:** "
            f"{count} tickets in the last 10 days"
        )


    # AI root cause analysis

    if st.button(
        "🔎 Analyze Potential Root Causes"
    ):

        if not client:

            st.error(
                "OpenAI API key not found."
            )

        else:

            with st.spinner(
                "Analyzing ticket patterns..."
            ):

                sample = df[
                    [
                        "created_at",
                        "product",
                        "ticket_subject",
                        "ticket_body",
                        "status"
                    ]
                ].to_string(
                    index=False
                )

                prompt = f"""
You are an operations analyst.

Analyze this support ticket dataset.

Identify:
1. The most important recurring issues
2. Any recent spikes or emerging issues
3. Potential root causes
4. Which issues Operations should investigate first

Important:
- Clearly distinguish observations from hypotheses.
- Do not claim a root cause is confirmed unless the data proves it.

Dataset:

{sample}
"""

                response = client.responses.create(
                    model="gpt-5.6-luna",
                    input=prompt
                )

                analysis = response.output_text

            st.subheader(
                "AI Root-Cause Analysis"
            )

            st.write(analysis)


# -----------------------------
# FOOTER
# -----------------------------

st.divider()

st.caption(
    "AI Support Operations Prototype | "
    "Built with Streamlit"
)