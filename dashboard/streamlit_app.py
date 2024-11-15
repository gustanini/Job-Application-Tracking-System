import streamlit as st
from health_check import health_check

def main():
    st.title("JATS Maintenance Dashboard")

    # Perform health check
    checks = health_check()
    st.subheader("Health Check Results")
    for check, status in checks.items():
        st.write(f"{check}: {'✅' if status else '❌'}")

    # Display logs
    st.subheader("Application Logs")
    try:
        with open("logs/jats.log", "r") as log_file:
            st.text(log_file.read())
    except FileNotFoundError:
        st.write("No logs available.")

if __name__ == "__main__":
    main()
