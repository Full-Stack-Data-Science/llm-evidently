import json
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Request
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
import logging
from evidently.future.datasets import Dataset, DataDefinition
from evidently.future.descriptors import TextLength, DeclineLLMEval, Sentiment
from prometheus_client import start_http_server

# Start Prometheus client
start_http_server(port=8099, addr="0.0.0.0")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up OpenTelemetry metrics
resource = Resource(attributes={SERVICE_NAME: "llm-monitor"})
reader = PrometheusMetricReader()
provider = MeterProvider(resource=resource, metric_readers=[reader])
set_meter_provider(provider)
meter = metrics.get_meter("llm_monitor", "0.1.0")

# Create metrics instruments
denial_counter = meter.create_counter(
    name="llm_denials_total",
    description="Total number of denied responses",
    unit="1"
)
response_counter = meter.create_counter(
    name="llm_responses_total",
    description="Total number of processed responses",
    unit="1"
)
response_length_histogram = meter.create_histogram(
    name="llm_response_length_chars",
    description="Distribution of response lengths in characters",
    unit="characters"
)

app = FastAPI()

def process_interaction(query: str, response: str):
    """Process single interaction and return metrics"""
    df = pd.DataFrame({
        "question": [query],
        "answer": [response]
    })
    
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
    return scored_df.iloc[0]

def log_llm_interaction(query, response, metrics):
    """Log interaction with calculated metrics"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "metrics": {
            "length": metrics["Length"],
            "denial": metrics["Denials"],
            "sentiment": metrics["Sentiment"]
        }
    }
    with open("llm_logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    logging.info(f"Processed interaction: {query[:30]}...")

@app.post("/generate")
async def generate_response(request: Request):
    data = await request.json()
    query = data.get("query")
    
    if not query:
        return {"error": "Missing query parameter"}
    
    # Generate response - REPLACE THIS WITH YOUR ACTUAL LLM CALL
    # For testing purposes, we'll create a mock response
    response = "This is a sample response to: " + query
    
    # Process metrics
    try:
        metrics = process_interaction(query, response)
        is_denial = metrics["Denials"] == "DECLINE"
        length = int(metrics["Length"])
        sentiment = float(metrics["Sentiment"])
        
        # Update metrics
        response_counter.add(1)
        if is_denial:
            denial_counter.add(1)
        response_length_histogram.record(length)
        
        # Log interaction
        log_llm_interaction(query, response, {
            "Length": length,
            "Denials": "DECLINE" if is_denial else "OK",
            "Sentiment": sentiment
        })
        
        return {
            "generated_response": response,
            "metrics": {
                "denial": is_denial,
                "length": length,
                "sentiment": sentiment
            }
        }
    
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return {"error": "Metrics processing failed", "details": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)