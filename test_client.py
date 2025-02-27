#!/usr/bin/env python3
"""
Test client for the Model Server API
"""

import argparse
import asyncio
import httpx
import json
import time
from typing import Dict, List, Any


class ModelServerClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(base_url=base_url, headers={"X-API-Key": api_key})

    async def health_check(self) -> Dict[str, Any]:
        """Check if the server is running."""
        response = await self.client.get("/health")
        return response.json()

    async def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        response = await self.client.get("/metrics")
        return response.json()

    async def generate_completion(self, prompt: str, system_message: str = None, max_tokens: int = None, temperature: float = 0.1) -> str:
        """Generate text completion."""
        payload = {"prompt": prompt, "system_message": system_message, "max_tokens": max_tokens, "temperature": temperature}

        response = await self.client.post("/v1/completions", json=payload)
        result = response.json()
        return result["text"]

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        payload = {"input": texts}

        response = await self.client.post("/v1/embeddings", json=payload)
        result = response.json()
        return result["embeddings"]

    async def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text into categories."""
        payload = {"text": text, "categories": categories}

        response = await self.client.post("/v1/classifications", json=payload)
        result = response.json()
        return result["classifications"]

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def run_tests(args):
    client = ModelServerClient(args.url, args.api_key)

    try:
        # Basic health check
        print("\nRunning health check...")
        health = await client.health_check()
        print(f"Health check result: {health}")

        # Test primary model
        if args.test_primary:
            print("\nTesting primary model...")
            start_time = time.time()
            completion = await client.generate_completion(
                prompt="Extract the job title, company name, and location from this job listing: 'Senior Software Engineer at Google in Mountain View, CA'",
                system_message="You are an expert job data extractor.",
            )
            elapsed = time.time() - start_time
            print(f"Completion generated in {elapsed:.2f}s:")
            print(completion)

        # Test embedding model
        if args.test_embedding:
            print("\nTesting embedding model...")
            texts = ["Software Engineer with experience in Python and JavaScript", "Data Scientist with experience in machine learning and statistics"]
            start_time = time.time()
            embeddings = await client.generate_embeddings(texts)
            elapsed = time.time() - start_time
            print(f"Embeddings generated in {elapsed:.2f}s:")
            print(f"Number of embeddings: {len(embeddings)}")
            print(f"Embedding dimensions: {len(embeddings[0])}")

        # Test classifier model
        if args.test_classifier:
            print("\nTesting classifier model...")
            text = "Looking for a software engineer with 5 years of experience in Python and JavaScript"
            categories = ["job_posting", "resume", "cover_letter", "job_application"]
            start_time = time.time()
            classifications = await client.classify_text(text, categories)
            elapsed = time.time() - start_time
            print(f"Classification completed in {elapsed:.2f}s:")
            print(json.dumps(classifications, indent=2))

        # Get metrics
        if args.metrics:
            print("\nGetting server metrics...")
            metrics = await client.get_metrics()
            print(json.dumps(metrics, indent=2))

    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test client for the Model Server API")
    parser.add_argument("--url", default="http://localhost:8080", help="Base URL of the model server")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--test-primary", action="store_true", help="Test primary model")
    parser.add_argument("--test-embedding", action="store_true", help="Test embedding model")
    parser.add_argument("--test-classifier", action="store_true", help="Test classifier model")
    parser.add_argument("--metrics", action="store_true", help="Get server metrics")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    # If --all flag is used, enable all tests
    if args.all:
        args.test_primary = True
        args.test_embedding = True
        args.test_classifier = True
        args.metrics = True

    # Run all tests
    asyncio.run(run_tests(args))
