import json
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
import threading
import logging
from evidently.future.datasets import Dataset, DataDefinition
from evidently.future.descriptors import TextLength, DeclineLLMEval, Sentiment
from evidently.future.report import Report
from evidently.future.presets import TextEvals
from prometheus_client import start_http_server


# Start Prometheus client
start_http_server(port=8099, addr="0.0.0.0") # Expose metrics on port 8099
 
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



# Global variables to hold current metric values
current_denial_rate = 0.0
current_avg_length = 0.0

# Set up OpenTelemetry for metrics
resource = Resource(attributes={SERVICE_NAME: "llm-monitor"})
reader = PrometheusMetricReader() 
provider = MeterProvider(resource=resource, metric_readers=[reader])
metrics.set_meter_provider(provider)
meter = metrics.get_meter("llm_monitor", "0.1.0")

# Define callbacks for ObservableGauge instruments
def denial_rate_callback():
    return [metrics.Observation(current_denial_rate)]

def avg_length_callback():
    return [metrics.Observation(current_avg_length)]

# Create ObservableGauge instruments
denial_rate_gauge = meter.create_observable_gauge(
    name="llm_denial_rate",
    description="Current denial rate of LLM responses",
    unit="1",
    callbacks=[denial_rate_callback]
)
avg_length_gauge = meter.create_observable_gauge(
    name="llm_avg_response_length",
    description="Average length of LLM responses",
    unit="characters",
    callbacks=[avg_length_callback]
)

# Initialize FastAPI app
app = FastAPI()

# Function to log LLM interactions
def log_llm_interaction(query, response, metadata=None):
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "query": query,
        "response": response,
        **(metadata or {})
    }
    with open("llm_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    logging.info(f"Logged interaction: {query[:30]}... -> {response[:30]}...")

# Function to evaluate logs and update metrics
def run_scheduled_evaluation():
    global current_denial_rate, current_avg_length
    try:
        logs = pd.read_json("llm_logs.jsonl", lines=True)
    except (FileNotFoundError, ValueError):
        logging.warning("No logs found or file is empty.")
        return

    one_day_ago = (datetime.now() - timedelta(days=1)).isoformat()
    recent_logs = logs[logs.timestamp > one_day_ago]

    if recent_logs.empty:
        logging.info("No recent logs to evaluate.")
        return

    df = pd.DataFrame({
        "question": recent_logs.query,
        "answer": recent_logs.response
    }).dropna(subset=["question", "answer"])

    if df.empty:
        logging.info("No valid data to evaluate after dropping NaNs.")
        return

    eval_dataset = Dataset.from_pandas(
        df,
        data_definition=DataDefinition(),
        descriptors=[
            TextLength("answer", alias="Length"),
            DeclineLLMEval("answer", alias="Denials", model="gpt-4o-mini"),
            Sentiment("answer", alias="Sentiment")
        ]
    )

    scored_df = eval_dataset.as_dataframe()
    report = Report([TextEvals()])
    my_eval = report.run(eval_dataset, None)
    report_file = f"llm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    my_eval.save_html(report_file)
    logging.info(f"Evaluation report saved to {report_file}.")

    # Calculate metrics
    denial_rate = (scored_df["Denials"] == "DECLINE").mean()
    avg_length = scored_df["Length"].mean()
    logging.info(f"Denial Rate: {denial_rate:.2%}, Avg Length: {avg_length:.1f}")

    # Update global metric values
    current_denial_rate = denial_rate
    current_avg_length = avg_length

    check_alert_thresholds(scored_df)

# Function to check thresholds and log alerts
def check_alert_thresholds(scored_df):
    thresholds = {
        "avg_denials": 0.10,  # Max 10% denial rate
        "avg_length": 50      # Min 50 characters average length
    }
    alerts = []
    denial_rate = (scored_df["Denials"] == "DECLINE").mean()
    if denial_rate > thresholds["avg_denials"]:
        alerts.append(f"High denial rate detected: {denial_rate:.2%}")
    avg_length = scored_df["Length"].mean()
    if pd.notna(avg_length) and avg_length < thresholds["avg_length"]:
        alerts.append(f"Low average response length: {avg_length:.1f} characters")
    if alerts:
        logging.warning("Alerts:\n" + "\n".join(alerts))
    else:
        logging.info("No issues detected.")

# Schedule periodic evaluation
def schedule_evaluation():
    run_scheduled_evaluation()
    threading.Timer(60, schedule_evaluation).start()  # Run every 60 seconds

# Start the scheduler
schedule_evaluation()

# FastAPI endpoint to log interactions
@app.post("/log_interaction")
async def log_interaction(request: Request):
    data = await request.json()
    query = data.get("query")
    response = data.get("response")
    if query and response:
        log_llm_interaction(query, response)
        return {"status": "logged"}
    else:
        return {"status": "error", "message": "Missing query or response"}

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)