# tests/benchmark_routing.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
from src.creditsystem.intent import classify_intent
from src.creditsystem.rag import retrieve_context
from src.creditsystem.decision import decide_execution
from src.creditsystem.handlers import route_to_handler
from src.creditsystem.cloud import handle_cloud_execution

TEST_CASES = [
    ("What is a credit score?",           "explanation",     "local"),
    ("How do I improve my credit score?", "advisory",        "local"),
    ("Is a CCJ bad for my credit?",       "risk_assessment", "local"),
    ("What if I miss 3 payments?",        "simulation",      "cloud"),
]

def test_intent_classification():
    """Test that intent classifier returns correct labels."""
    for query, expected_intent, _ in TEST_CASES:
        intent = classify_intent(query)
        assert intent == expected_intent, (
            f"Query: '{query}'\n"
            f"Expected intent: {expected_intent}\n"
            f"Got: {intent}"
        )

def test_routing_decision():
    """Test that routing correctly sends queries local vs cloud."""
    for query, expected_intent, expected_route in TEST_CASES:
        intent = classify_intent(query)
        context, confidence = retrieve_context(query, intent)
        decision = decide_execution(intent, confidence)
        assert decision == expected_route, (
            f"Query: '{query}'\n"
            f"Expected route: {expected_route}\n"
            f"Got: {decision} (confidence={confidence:.3f})"
        )

def test_confidence_is_valid():
    """Test that confidence scores are between 0 and 1."""
    for query, expected_intent, _ in TEST_CASES:
        intent = classify_intent(query)
        context, confidence = retrieve_context(query, intent)
        assert 0.0 <= confidence <= 1.0, (
            f"Confidence out of range for '{query}': {confidence}"
        )

def test_local_returns_response():
    """Test that local handler returns a non-empty response."""
    query = "What is a credit score?"
    intent = classify_intent(query)
    context, confidence = retrieve_context(query, intent)
    response = route_to_handler(intent, query, context)
    assert response is not None
    assert len(response) > 50, "Response too short, something went wrong"

def test_latency_local_faster_than_cloud():
    """Test that local responses are faster than cloud on average."""
    local_times = []
    cloud_times = []

    for query, _, expected_route in TEST_CASES:
        intent = classify_intent(query)
        context, confidence = retrieve_context(query, intent)
        
        t0 = time.time()
        if expected_route == "local":
            route_to_handler(intent, query, context)
            local_times.append(time.time() - t0)
        else:
            handle_cloud_execution(query, intent)
            cloud_times.append(time.time() - t0)

    if local_times and cloud_times:
        avg_local = sum(local_times) / len(local_times)
        avg_cloud = sum(cloud_times) / len(cloud_times)
        print(f"\nAvg local: {avg_local:.2f}s | Avg cloud: {avg_cloud:.2f}s")
        assert avg_local < avg_cloud, (
            f"Expected local ({avg_local:.2f}s) to be faster than cloud ({avg_cloud:.2f}s)"
        )