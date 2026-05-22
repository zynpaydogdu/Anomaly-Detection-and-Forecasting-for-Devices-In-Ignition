# Industrial Anomaly Detection and Forecasting Prototype

An end-to-end industrial anomaly detection and forecasting prototype built with **Ignition Perspective**, **Ignition WebDev**, **FastAPI**, and **PyTorch**.

This project monitors simulated IIoT sensor data, detects abnormal behavior, calculates anomaly scores, compares actual values with predicted behavior, and visualizes the results on a real-time dashboard.
> **Note:** This project uses simulated IIoT sensor data. It does not contain real production data or confidential company information.

---

## Project Overview

The main goal of this project is to demonstrate how industrial sensor data can be monitored through an SCADA-like dashboard and analyzed by a machine learning backend.

The system collects historical tag values from Ignition, sends them to a Python backend, trains or loads deep learning models, performs inference, and returns anomaly detection results back to the dashboard.

The dashboard displays:

- Actual sensor data

- Predicted sensor behavior

- Anomaly score

- Threshold line

- Detected anomaly points

- Batch-based inference results

- User-friendly status and error messages

---

## Architecture

```text

Ignition Perspective Dashboard

        |

        | HTTP POST

        v

Ignition WebDev Endpoints

        |

        | Tag History Query

        v

Ignition Project Scripts

        |

        | JSON Request

        v

FastAPI Backend

        |

        | PyTorch Models

        v

LSTM Forecasting + LSTM Autoencoder Anomaly Detection

        |

        | JSON Response

        v

Ignition Dashboard Visualization

Main Components

Component

Description

Ignition Perspective

User interface and dashboard visualization

Ignition WebDev

Middleware layer between Perspective and FastAPI

Ignition Tag History

Historical simulated sensor data source

FastAPI

Python REST API backend

PyTorch

Deep learning model development and inference

LSTM

Forecasting expected sensor behavior

LSTM Autoencoder

Detecting abnormal behavior using reconstruction error


⸻


Technologies Used

	•	Python

	•	FastAPI

	•	PyTorch

	•	NumPy

	•	Joblib

	•	Pydantic

	•	Uvicorn

	•	Ignition Perspective

	•	Ignition WebDev

	•	Ignition Tag History

	•	REST API

	•	JSON

	•	GitHub


⸻


Machine Learning Approach

LSTM Forecasting

The LSTM forecasting model learns the normal time-dependent behavior of simulated sensor values. It predicts the expected sensor behavior based on previous time steps.

This prediction is displayed on the dashboard as Predicted Data, while the original sensor values are displayed as Actual Data.

LSTM Autoencoder for Anomaly Detection

The Autoencoder learns the normal behavior pattern of the sensor data.

It works as follows:

	1.	The encoder compresses the input time-series window into a latent representation.

	2.	The decoder tries to reconstruct the original input sequence.

	3.	The difference between the real value and the reconstructed value is calculated as reconstruction error.

	4.	This reconstruction error is used as the anomaly score.

	5.	If the anomaly score exceeds the threshold, the point is marked as an anomaly.


⸻


Threshold Logic

The threshold is not a direct limit on the sensor value. It is a decision boundary for the model’s reconstruction error.

In this project, the threshold is calculated from normal reconstruction error values. The system calculates two statistical thresholds:

mean + 3 * standard deviation

and

Q3 + 1.5 * IQR

The smaller value is selected as the final threshold to make anomaly detection more sensitive.

If the anomaly score is greater than the threshold, the system marks the point as an anomaly.


⸻


Features

	•	Simulated IIoT sensor monitoring

	•	Ignition Perspective dashboard

	•	WebDev-based middleware endpoints

	•	FastAPI backend integration

	•	REST API request-response flow

	•	LSTM-based forecasting

	•	LSTM Autoencoder-based anomaly detection

	•	Reconstruction error-based anomaly scoring

	•	Dynamic threshold calculation

	•	Batch inference support

	•	Auto Refresh support

	•	Error handling for:

	◦	Backend unavailable

	◦	Not enough data

	◦	Empty/null data

	◦	Constant/flat sensor data

	◦	JSON response errors

	◦	Authentication errors

	•	User-friendly popup and status card feedback

	•	Gateway logging for debugging


⸻


## Repository Structure
Anomaly-Detection-and-Forecasting-for-Devices-In-Ignition/
│
├── backend/
│   ├── main.py
│   ├── models.py
│   ├── router_infer.py
│   ├── router_train.py
│   ├── schemas.py
│   └── utils.py
│
└── Ignition/
   ├── perspective/
   │   └── com.inductiveautomation.perspective/
   │       └── views/
   │           └── Perspective view files for the dashboard interface
   │
   ├── scripts/
   │   └── ignition/
   │       └── Ignition project scripts such as anomaly communication logic
   │
   └── webdev/
       └── com.inductiveautomation.webdev/
           └── resources/
               └── ad/
                   └── WebDev endpoint resources for train and infer requests

### Folder Descriptions

- `backend/` contains the Python FastAPI backend. It includes the API entry point, training and inference routers, PyTorch model definitions, request schemas, and utility functions.

- `Ignition/perspective/` contains the Ignition Perspective dashboard view files.

- `Ignition/scripts/` contains Ignition project scripts used for collecting tag history data and communicating with the backend.

- `Ignition/webdev/` contains Ignition WebDev resources used as middleware endpoints between the Perspective dashboard and the FastAPI backend.### Folder Descriptions

- `backend/` contains the Python FastAPI backend. It includes the API entry point, training and inference routers, PyTorch model definitions, request schemas, and utility functions.

- `Ignition/perspective/` contains the Ignition Perspective dashboard view files.

- `Ignition/scripts/` contains Ignition project scripts used for collecting tag history data and communicating with the backend.

- `Ignition/webdev/` contains Ignition WebDev resources used as middleware endpoints between the Perspective dashboard and the FastAPI backend.
 

⸻


Backend Setup

1. Create a virtual environment

python -m venv .venv

Activate it:

.venv\Scripts\activate

For Linux/macOS:

source .venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

Example requirements.txt:

fastapi

uvicorn

numpy

torch

joblib

pydantic

3. Run the FastAPI backend

python -m uvicorn main:app --host 0.0.0.0 --port 8000

The backend should be available at:
http://localhost:8000

Health check endpoint:

GET http://localhost:8000/health

Expected response:

{

  "status": "ok"

}


⸻


FastAPI Endpoints

/train

Used to train or load existing models for selected sensor tags.

POST /train

Example request:

{

  "job_id": "ignition-auto-train",

  "series": [

    {

      "tag": "[default]test/AnomalyTest1",

      "timestamps": [

        "2026-05-06T10:00:00.000",

        "2026-05-06T10:00:05.000"

      ],

      "values": [200.1, 201.3]

    }

  ],

  "meta": {

    "epochs": 20,

    "tuning": true,

    "force_retrain": false

  }

}

/infer

Used to perform forecasting and anomaly detection.

POST /infer

Example request:

{

  "request_id": "ignition-infer",

  "series": [

    {

      "tag": "[default]test/AnomalyTest1",

      "timestamps": [

        "2026-05-06T10:00:00.000",

        "2026-05-06T10:00:05.000"

      ],

      "values": [200.1, 201.3]

    }

  ],

  "meta": {

    "holdout_minutes": 10,

    "step_seconds": 5,

    "batch_minutes": 10,

    "batch_stride_minutes": 10,

    "anomaly_confirm_points": 1,

    "recover_points": 6,

    "range_margin": 0.03

  }

}


⸻


Ignition Setup

1. Import the Ignition Project

	1.	Open Ignition Designer.

	2.	Import the project export file from:

ignition/export/

	3.	Update local paths, backend URL, and tag providers if needed.

2. Configure Simulated Tags

The project expects simulated sensor tags under a path similar to:

[default]test/AnomalyTest1

[default]test/AnomalyTest2

...

These tags are used as simulated IIoT sensor data sources.

3. Enable Tag History

Tag History should be enabled for the simulated sensor tags. The WebDev layer reads historical values using:

system.tag.queryTagHistory(...)

4. Configure WebDev Endpoints

The project uses two WebDev endpoints:

/system/webdev/staj_anomalydetect/ad/train

/system/webdev/staj_anomalydetect/ad/infer

These endpoints receive requests from Perspective, collect Tag History data, and forward JSON requests to the FastAPI backend.


⸻


Security Configuration

WebDev endpoints can be protected using Ignition WebDev security settings.

Recommended configuration:

Require Authentication: enabled

User Source: default

Required Role(s): AnomalyWebDev

Max Auth Retries: 3

A dedicated service account can be created for internal WebDev calls:

Username: svc_anomaly_webdev

Role: AnomalyWebDev

Credentials should not be hardcoded directly in the button script. Use a separate configuration file or secure configuration mechanism.


Example file:

# security_config_example.py

def getWebDevAuth():

    return {

        "username": "YOUR_SERVICE_ACCOUNT_USERNAME",

        "password": "YOUR_SERVICE_ACCOUNT_PASSWORD"

    }

Do not commit real credentials to GitHub.


⸻


Dashboard Usage

	1.	Start the FastAPI backend.

	2.	Open the Ignition Perspective dashboard.

	3.	Select a simulated sensor tag.

	4.	Click the check button to start training and inference.

	5.	View:

	◦	Actual vs predicted graph

	◦	Anomaly score graph

	◦	Threshold line

	◦	Detected anomaly table

	◦	Status card

	◦	Popup messages

If Auto Refresh is enabled, the dashboard periodically calls the inference endpoint and updates the charts.


⸻


Error Handling

The project includes user-facing error handling for common failure scenarios.

Scenario

User Message

Backend is not running

FastAPI server could not be reached

Not enough data

There is no enough data for training

Empty or flat data

Sensor data is constant or invalid

Authentication failure

WebDev authentication failed

Authorization failure

User does not have required role

JSON parse issue

WebDev response error

Detailed logs are written to Ignition Gateway Logs using custom logger names such as:

IF-API

WEBDEV-TRAIN

WEBDEV-INFER

BTN-WEBDEV

INFER-REFRESH


⸻


Notes About Data

This project does not use real production data.

The data used in this project is simulated IIoT sensor data generated inside Ignition. This allows testing of forecasting, anomaly detection, fault injection, dashboard behavior, and error handling without exposing confidential industrial data.


⸻


Limitations

This project is a prototype and is not production-ready without additional improvements.

Possible improvements:

	•	Testing with real industrial sensor data

	•	More advanced authentication and secret management

	•	Model versioning

	•	Better model lifecycle management

	•	Scheduled retraining strategy

	•	More robust monitoring and alerting

	•	Docker-based deployment

	•	External database integration

	•	More advanced anomaly validation methods

	•	Role-based UI permissions


⸻


Possible Future Improvements

	•	Add notification or alarm integration

	•	Add model performance monitoring

	•	Add advanced fault injection scenarios

	•	Store inference results in a database

	•	Add dashboard-level user management

	•	Add Docker Compose deployment

	•	Add CI/CD pipeline

	•	Add real-time streaming support


⸻


Disclaimer

This repository is intended for educational and prototype demonstration purposes. It uses simulated IIoT data and does not include confidential company data, real production data, or internal infrastructure details.
