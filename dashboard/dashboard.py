# dashboard.py
"""
This module handles dashboard-related code and templates.
"""
import streamlit as st

def maintenance_dashboard(checks):
    """
    Display the system health and logs in a dashboard.
    Args:
        checks (dict): Results from health_check.
    """
    st.title("Maintenance Dashboard")
    st.subheader("Health Check Results")
    for check, status in checks.items():
        st.write(f"{check}: {'✅' if status else '❌'}")
    st.subheader("Recent Logs")
    with open("logs/jats.log", "r") as log_file:
        st.text(log_file.read())
