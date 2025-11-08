##  DAG Flow

1. OCR text extraction from uploaded or cloud-stored loan documents.
2. Vector index creation for semantic retrieval.
3. Prompt generation and LLM query execution through the Ollama API.
4. Robust logging, anomaly tracking, and end-to-end monitoring.

##  Environment Setup and Configuration

### Build Docker Environment

Before initializing Airflow, ensure Docker and Docker Compose are installed. Then build the environment:

```bash
docker-compose down -v
docker-compose up -d --build
```

This builds the custom Airflow image defined in `Dockerfile.airflow` and starts all services:

* `airflow-webserver`
* `airflow-scheduler`
* `postgres`
* `redis`
* `ollama` (running locally on port `11434`)


###  Verify Container Status

```bash
docker ps
```

Confirm that the following containers are running:

```
dl_project-airflow-webserver-1
dl_project-airflow-scheduler-1
dl_project-redis-1
dl_project-postgres-1
```

---

###  Access the Airflow UI

Open:

```
http://localhost:8080
```

If you encounter **“Invalid login”**, reset the Airflow admin account:

```bash
docker exec -it dl_project-airflow-webserver-1 airflow users create \
--username admin \
--password admin \
--firstname Admin \
--lastname User \
--role Admin \
--email admin@example.com
```


###  DAG Path Verification

Confirm that Airflow detects your DAG file:

```bash
docker exec dl_project-airflow-scheduler-1 airflow dags list | grep loan_doc_pipeline_dag
```

Expected output:

```
loan_doc_pipeline_dag | /opt/airflow/dags/loan_doc_pipeline_dag.py | username | True
```

If `True` → the DAG is paused initially.

Unpause it:

```bash
docker exec -it dl_project-airflow-scheduler-1 airflow dags unpause loan_doc_pipeline_dag
```


## Running and Monitoring the DAG

### Trigger DAG Manually

```bash
docker exec -it dl_project-airflow-scheduler-1 airflow dags trigger loan_doc_pipeline_dag --run-id manual_run_$(date +%Y%m%d_%H%M%S)
```

This starts a full pipeline run.


###  Check DAG Run Status

```bash
docker exec -it dl_project-airflow-scheduler-1 airflow dags list-runs -d loan_doc_pipeline_dag
```


###  View Task-Level States

```bash
docker exec -it dl_project-airflow-scheduler-1 airflow tasks states-for-dag-run loan_doc_pipeline_dag <run_id>
```

Example:

```
loan_doc_pipeline_dag | 2025-11-07T21:55:17+00:00 | extract_text        | success |
loan_doc_pipeline_dag | 2025-11-07T21:55:17+00:00 | update_vector_index | success |
loan_doc_pipeline_dag | 2025-11-07T21:55:17+00:00 | generate_llm_prompt | running |
```


###  View Live Logs

**From Airflow Scheduler:**

```bash
docker exec -it dl_project-airflow-scheduler-1 bash -c "tail -f /opt/airflow/logs/dag_id=loan_doc_pipeline_dag/run_id=<run_id>/task_id=generate_llm_prompt/attempt=1.log"
```

**From the centralized logs directory:**

```bash
docker exec -it dl_project-airflow-scheduler-1 bash -c "cat /opt/airflow/logs/llm_logs/llm_<timestamp>.log | tail -n 40"
```

**For anomalies:**

```bash
docker exec -it dl_project-airflow-scheduler-1 bash -c "tail -f /opt/airflow/logs/anomaly_logs/anomaly_<timestamp>.log"
```


## DAG File Structure

### `loan_doc_pipeline_dag.py`

Main orchestration DAG defining 3 core tasks:

| Task                  | Function                            | Description                                                      |
| --------------------- | ----------------------------------- | ---------------------------------------------------------------- |
| `extract_text`        | `process_single_file()`             | Runs OCR and preprocessing for all uploaded PDFs/images.         |
| `update_vector_index` | `add_to_index()`                    | Updates vector embeddings and semantic index for retrieval.      |
| `generate_llm_prompt` | `build_prompt()` & `query_ollama()` | Routes queries through LLM prompt engine and executes inference. |

Each stage logs both **normal activity** and **anomaly events** to `/opt/airflow/logs`.

---

## Logging and Anomaly Tracking

Centralized logging architecture under `/opt/airflow/logs`:

```
logs/
├── extraction_logs/        # OCR and text preprocessing logs
├── llm_logs/               # LLM prompt + response logs
├── anomaly_logs/           # Detected anomalies (timeouts, low confidence)
├── dag_logs/               # DAG orchestration logs
├── scheduler/              # Scheduler process output
└── dag_processor_manager/  # Airflow DAG processor logs
```

### Common log commands:

```bash
docker exec -it dl_project-airflow-scheduler-1 bash -c "tail -f /opt/airflow/logs/llm_logs/llm_*.log"
docker exec -it dl_project-airflow-scheduler-1 bash -c "tail -f /opt/airflow/logs/anomaly_logs/anomaly_*.log"
```


## LLM Integration and Verification

### Check Ollama Connectivity

Inside the Airflow container:

```bash
docker exec -it dl_project-airflow-scheduler-1 curl http://host.docker.internal:11434/api/tags
```

Expected models:

```
phi3, mistral, wizard-math, llama3
```

---

### Test Ollama Response Manually

```bash
docker exec -it dl_project-airflow-scheduler-1 python3 - <<'PY'
from scripts.LLMquery.prompts.prompt_router import query_ollama
out = query_ollama("Say hello from Airflow pipeline", model="phi3")
print("\n--- Ollama Output ---\n", out[:400])
PY
```
 
Expected:

```
Ollama model 'phi3' responded successfully (...)
Hello! This is a response from the Airflow-integrated LLM pipeline...
```

---

##  Common Issues & Fixes

| Issue                                                        | Cause                               | Fix                                                                                   |
| ------------------------------------------------------------ | ----------------------------------- | ------------------------------------------------------------------------------------- |
|   *Invalid Login*                                            | Airflow credentials missing         | Recreate admin user using `airflow users create ...`                                  |
|    *Ollama Timeout*                                          | Model server not reachable          | Ensure Ollama app is running locally: `ollama serve`                                  |
|    *generate_llm_prompt stuck*                               | LLM taking too long or unresponsive | Extend timeout or reduce context size in `prompt_router.py`                           |
|    *DAG not visible*                                         | Wrong mount path or paused          | Rebuild containers and unpause DAG using `airflow dags unpause loan_doc_pipeline_dag` |
|    *Decoding failed: entrypoint invalid command line string* | YAML format error                   | Ensure `docker-compose.yaml` uses proper multiline syntax for entrypoints             |


## Validation Commands Summary

| Purpose        | Command                                                                                                                                |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| List all DAGs  | `docker exec -it dl_project-airflow-scheduler-1 airflow dags list`                                                                     |
| Unpause DAG    | `docker exec -it dl_project-airflow-scheduler-1 airflow dags unpause loan_doc_pipeline_dag`                                            |
| Trigger DAG    | `docker exec -it dl_project-airflow-scheduler-1 airflow dags trigger loan_doc_pipeline_dag --run-id manual_run_$(date +%Y%m%d_%H%M%S)` |
| Check DAG runs | `docker exec -it dl_project-airflow-scheduler-1 airflow dags list-runs -d loan_doc_pipeline_dag`                                       |
| Task states    | `docker exec -it dl_project-airflow-scheduler-1 airflow tasks states-for-dag-run loan_doc_pipeline_dag <run_id>`                       |
| View logs      | `docker exec -it dl_project-airflow-scheduler-1 bash -c "tail -f /opt/airflow/logs/..."`                                               |


## Restarting from Scratch

If you need to reset the environment completely:

```bash
docker-compose down -v
docker system prune -f
docker-compose up -d --build
```

Then re-trigger the DAG:

```bash
docker exec -it dl_project-airflow-scheduler-1 airflow dags trigger loan_doc_pipeline_dag --run-id manual_run_$(date +%Y%m%d_%H%M%S)
```
