# Streamlit Frontend Components

This directory contains reusable components for the Streamlit frontend application.

## API Client

The `api_client.py` module provides a client for communicating with the Product Hunt RAG Analyzer backend API.

### Features

-   **Health Checking**: Verify backend connectivity and service status
-   **Dataset Statistics**: Retrieve information about loaded indices
-   **Analysis Submission**: Submit product ideas for competitive analysis
-   **Error Handling**: Comprehensive error handling for network failures, timeouts, and API errors
-   **Context Manager Support**: Use as a context manager for automatic resource cleanup

### Usage

```python
from streamlit_app.components.api_client import APIClient

# Create client
client = APIClient(base_url="http://localhost:8000", timeout=300)

# Check if backend is available
if client.is_backend_available():
    # Get health status
    health = client.check_health()
    print(f"Status: {health['status']}")

    # Get dataset statistics
    stats = client.get_dataset_stats()
    print(f"Products: {stats['total_products']}")

    # Submit analysis
    result = client.submit_analysis(
        product_idea="A task management app",
        max_competitors=5,
        output_format="json"
    )
    print(f"Analysis ID: {result['analysis_id']}")
```

### Using as Context Manager

```python
with APIClient() as client:
    health = client.check_health()
    print(f"Status: {health['status']}")
```

### Error Handling

The API client raises specific exceptions for different error scenarios:

-   `APIConnectionError`: Unable to connect to backend
-   `APITimeoutError`: Request timed out
-   `APIResponseError`: Backend returned an error response
-   `APIClientError`: Generic client error

```python
from streamlit_app.components.api_client import (
    APIClient,
    APIConnectionError,
    APITimeoutError,
    APIResponseError
)

try:
    client = APIClient()
    result = client.submit_analysis("My product idea")
except APIConnectionError as e:
    print(f"Cannot connect to backend: {e}")
except APITimeoutError as e:
    print(f"Request timed out: {e}")
except APIResponseError as e:
    print(f"API error (status {e.status_code}): {e}")
```

### Configuration

The API client can be configured via:

1. **Constructor parameters**:

    ```python
    client = APIClient(base_url="http://example.com:8000", timeout=60)
    ```

2. **Environment variables**:
    ```bash
    export BACKEND_URL=http://example.com:8000
    ```

### Testing

Run unit tests:

```bash
pytest streamlit_app/tests/test_api_client.py -v
```

Run integration tests (requires running backend):

```bash
pytest streamlit_app/tests/test_api_client_integration.py -v
```

### Example

See `streamlit_app/examples/api_client_example.py` for a complete usage example.
