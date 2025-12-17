"""
Example usage of the State Manager module.

This example demonstrates how to use the state management functions
in a Streamlit application.
"""

import streamlit as st
from streamlit_app.components import (
    initialize_state,
    reset_analysis_state,
    update_config_state,
    preserve_config_on_reset
)


def main():
    """Main example application."""
    st.title("State Manager Example")
    
    # Initialize state on first load
    initialize_state()
    
    st.header("Current State")
    st.write("Analysis State:")
    st.json({
        "analysis_submitted": st.session_state.analysis_submitted,
        "analysis_running": st.session_state.analysis_running,
        "analysis_results": st.session_state.analysis_results,
        "analysis_error": st.session_state.analysis_error
    })
    
    st.write("Configuration State:")
    st.json({
        "product_idea": st.session_state.product_idea,
        "max_competitors": st.session_state.max_competitors,
        "output_format": st.session_state.output_format
    })
    
    st.write("Backend State:")
    st.json({
        "backend_connected": st.session_state.backend_connected,
        "dataset_stats": st.session_state.dataset_stats
    })
    
    st.header("State Management Actions")
    
    # Configuration updates
    st.subheader("Update Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        new_competitors = st.slider(
            "Max Competitors",
            min_value=1,
            max_value=10,
            value=st.session_state.max_competitors
        )
        if st.button("Update Competitors"):
            update_config_state(max_competitors=new_competitors)
            st.success(f"Updated max_competitors to {new_competitors}")
            st.rerun()
    
    with col2:
        new_format = st.selectbox(
            "Output Format",
            options=["json", "markdown", "pdf"],
            index=["json", "markdown", "pdf"].index(st.session_state.output_format)
        )
        if st.button("Update Format"):
            update_config_state(output_format=new_format)
            st.success(f"Updated output_format to {new_format}")
            st.rerun()
    
    # Simulate analysis
    st.subheader("Simulate Analysis")
    if st.button("Start Analysis"):
        st.session_state.analysis_submitted = True
        st.session_state.analysis_running = True
        st.session_state.analysis_results = {
            "competitors": ["Product A", "Product B"],
            "confidence": 0.85
        }
        st.success("Analysis completed!")
        st.rerun()
    
    # Reset analysis
    st.subheader("Reset Analysis")
    if st.button("Clear Results"):
        # Preserve config before reset
        saved_config = preserve_config_on_reset()
        st.info(f"Preserving config: {saved_config}")
        
        # Reset analysis state
        reset_analysis_state()
        st.success("Analysis state cleared! Configuration preserved.")
        st.rerun()


if __name__ == "__main__":
    main()
