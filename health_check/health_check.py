from logs.log_config import log_event
from data.data_loader import load_csv, load_cv
from config import CONFIG

def health_check():
    """
    Perform a health check of the application.
    Returns:
        dict: Status of critical components.
    """
    checks = {
        "CSV File Loaded": False,
        "CV File Loaded": False,
        "Logging Active": False,
    }

    # Check if the CSV and CV files can be loaded
    try:
        jobs_data = load_csv(CONFIG["CSV_PATH"])
        if jobs_data is not None:
            checks["CSV File Loaded"] = True
    except Exception as e:
        log_event(f"Health check failed: Unable to load CSV - {e}", level="error")

    try:
        cv_data = load_cv(CONFIG["CV_PATH"])
        if cv_data:
            checks["CV File Loaded"] = True
    except Exception as e:
        log_event(f"Health check failed: Unable to load CV - {e}", level="error")

    # Check if logging is active
    try:
        log_event("Health check: Logging is active")
        checks["Logging Active"] = True
        print("Logging is active!") # message
    except Exception as e:
        log_event(f"Health check failed: Logging error - {e}", level="error")

    return checks
