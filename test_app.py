#!/usr/bin/env python3
"""
Test script for the Custom Detection API
"""

import asyncio
import requests
import time
import json
import logging
from pathlib import Path
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "test_image.jpg"


def create_test_image():
    """Create a test image for testing"""
    # Create a simple test image
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Add some shapes to simulate ingredients/nutrition tables
    cv2.rectangle(img, (100, 100), (400, 300), (255, 255, 255), -1)
    cv2.rectangle(img, (120, 120), (380, 280), (0, 0, 0), 2)
    cv2.circle(img, (500, 500), 80, (255, 255, 255), -1)
    cv2.circle(img, (500, 500), 60, (0, 0, 0), 2)
    
    # Save test image
    cv2.imwrite(TEST_IMAGE_PATH, img)
    logger.info(f"Created test image: {TEST_IMAGE_PATH}")


def test_health_endpoint():
    """Test the health endpoint"""
    logger.info("Testing health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Health check passed: {data}")
            return True
        else:
            logger.error(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return False


def test_models_endpoint():
    """Test the models endpoint"""
    logger.info("Testing models endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            logger.info(f"Available models: {models}")
            return True
        else:
            logger.error(f"Models endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Models endpoint failed: {str(e)}")
        return False


def test_ingredient_detection():
    """Test ingredient detection"""
    logger.info("Testing ingredient detection...")
    
    try:
        with open(TEST_IMAGE_PATH, "rb") as f:
            files = {"file": f}
            params = {"confidence": 0.5, "iou": 0.45}
            response = requests.post(
                f"{BASE_URL}/detect/ingredient", 
                files=files, 
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                # Save the result
                with open("test_ingredient_result.jpg", "wb") as f:
                    f.write(response.content)
                logger.info("Ingredient detection test passed")
                return True
            else:
                logger.error(f"Ingredient detection failed with status {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"Ingredient detection failed: {str(e)}")
        return False


def test_nutrition_detection():
    """Test nutrition table detection"""
    logger.info("Testing nutrition table detection...")
    
    try:
        with open(TEST_IMAGE_PATH, "rb") as f:
            files = {"file": f}
            params = {"confidence": 0.5, "iou": 0.45}
            response = requests.post(
                f"{BASE_URL}/detect/nutrition", 
                files=files, 
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                # Save the result
                with open("test_nutrition_result.jpg", "wb") as f:
                    f.write(response.content)
                logger.info("Nutrition detection test passed")
                return True
            else:
                logger.error(f"Nutrition detection failed with status {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"Nutrition detection failed: {str(e)}")
        return False


def test_both_detection():
    """Test both ingredient and nutrition detection"""
    logger.info("Testing both detection...")
    
    try:
        with open(TEST_IMAGE_PATH, "rb") as f:
            files = {"file": f}
            params = {"confidence": 0.5, "iou": 0.45}
            response = requests.post(
                f"{BASE_URL}/detect/both", 
                files=files, 
                params=params,
                timeout=60
            )
            
            if response.status_code == 200:
                results = response.json()
                logger.info(f"Both detection test passed: {results}")
                return True
            else:
                logger.error(f"Both detection failed with status {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"Both detection failed: {str(e)}")
        return False


def test_json_detection():
    """Test JSON detection endpoint"""
    logger.info("Testing JSON detection...")
    
    try:
        with open(TEST_IMAGE_PATH, "rb") as f:
            files = {"file": f}
            params = {"model_type": "ingredient", "confidence": 0.5, "iou": 0.45}
            response = requests.post(
                f"{BASE_URL}/detect/json", 
                files=files, 
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                results = response.json()
                logger.info(f"JSON detection test passed: {results}")
                return True
            else:
                logger.error(f"JSON detection failed with status {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"JSON detection failed: {str(e)}")
        return False


def run_performance_test():
    """Run a simple performance test"""
    logger.info("Running performance test...")
    
    try:
        with open(TEST_IMAGE_PATH, "rb") as f:
            files = {"file": f}
            params = {"confidence": 0.5, "iou": 0.45}
            
            # Test ingredient detection performance
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/detect/ingredient", 
                files=files, 
                params=params,
                timeout=30
            )
            ingredient_time = time.time() - start_time
            
            # Test nutrition detection performance
            f.seek(0)  # Reset file pointer
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/detect/nutrition", 
                files=files, 
                params=params,
                timeout=30
            )
            nutrition_time = time.time() - start_time
            
            logger.info(f"Performance test results:")
            logger.info(f"  Ingredient detection: {ingredient_time:.3f}s")
            logger.info(f"  Nutrition detection: {nutrition_time:.3f}s")
            
            return True
    except Exception as e:
        logger.error(f"Performance test failed: {str(e)}")
        return False


def main():
    """Main test function"""
    logger.info("Starting Custom Detection API tests...")
    
    # Create test image
    create_test_image()
    
    # Run tests
    tests = [
        ("Health Endpoint", test_health_endpoint),
        ("Models Endpoint", test_models_endpoint),
        ("Ingredient Detection", test_ingredient_detection),
        ("Nutrition Detection", test_nutrition_detection),
        ("Both Detection", test_both_detection),
        ("JSON Detection", test_json_detection),
        ("Performance Test", run_performance_test),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The API is working correctly.")
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
    
    # Cleanup
    if Path(TEST_IMAGE_PATH).exists():
        Path(TEST_IMAGE_PATH).unlink()
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 