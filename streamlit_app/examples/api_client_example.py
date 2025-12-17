"""
Example usage of the API Client.

This script demonstrates how to use the APIClient to interact with
the Product Hunt RAG Analyzer backend.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from streamlit_app.components.api_client import (
    APIClient,
    APIConnectionError,
    APITimeoutError,
    APIResponseError
)


def main():
    """Demonstrate API client usage."""
    
    print("=" * 60)
    print("Product Hunt RAG Analyzer - API Client Example")
    print("=" * 60)
    print()
    
    # Create API client
    print("Creating API client...")
    client = APIClient(base_url="http://localhost:8000", timeout=300)
    print(f"✓ Client created with base URL: {client.base_url}")
    print()
    
    # Check backend availability
    print("Checking backend availability...")
    if client.is_backend_available():
        print("✓ Backend is available and responding")
    else:
        print("✗ Backend is not available")
        print("  Please ensure the backend is running:")
        print("  uvicorn src.api.app:app --host 0.0.0.0 --port 8000")
        return
    print()
    
    # Check health
    print("Checking backend health...")
    try:
        health = client.check_health()
        print(f"✓ Health check successful")
        print(f"  Status: {health['status']}")
        print(f"  Version: {health['version']}")
        print(f"  Ollama connected: {health['ollama_connected']}")
        print(f"  Indices loaded: {health['indices_loaded']}")
    except (APIConnectionError, APITimeoutError, APIResponseError) as e:
        print(f"✗ Health check failed: {e}")
        return
    print()
    
    # Get dataset stats
    print("Retrieving dataset statistics...")
    try:
        stats = client.get_dataset_stats()
        print(f"✓ Dataset stats retrieved")
        print(f"  Total products: {stats['total_products']}")
        print(f"  Total reviews: {stats['total_reviews']}")
        print(f"  Avg reviews per product: {stats['avg_reviews_per_product']}")
        print(f"  Indices loaded: {stats['indices_loaded']}")
    except (APIConnectionError, APITimeoutError, APIResponseError) as e:
        print(f"✗ Failed to retrieve stats: {e}")
    print()
    
    # Submit analysis (optional - commented out by default)
    print("To submit an analysis, uncomment the code below:")
    print()
    print("# Submit analysis")
    print("# product_idea = 'A task management app for remote teams'")
    print("# print(f'Submitting analysis for: {product_idea}')")
    print("# try:")
    print("#     result = client.submit_analysis(")
    print("#         product_idea=product_idea,")
    print("#         max_competitors=5,")
    print("#         output_format='json'")
    print("#     )")
    print("#     print(f'✓ Analysis completed')")
    print("#     print(f'  Analysis ID: {result[\"analysis_id\"]}')")
    print("#     print(f'  Competitors found: {len(result[\"competitors_identified\"])}')")
    print("#     print(f'  Confidence score: {result[\"confidence_score\"]:.2f}')")
    print("#     print(f'  Processing time: {result[\"processing_time_ms\"]}ms')")
    print("# except (APIConnectionError, APITimeoutError, APIResponseError) as e:")
    print("#     print(f'✗ Analysis failed: {e}')")
    print()
    
    # Example with context manager
    print("Example using context manager:")
    print()
    try:
        with APIClient() as client:
            health = client.check_health()
            print(f"✓ Health check via context manager: {health['status']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()
    
    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
