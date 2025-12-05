#!/usr/bin/env python
import warnings
from nearbyhospitals.crew import Nearbyhospitals

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the crew to fetch hospitals.
    """
    inputs = {
        "city": "gharaunda",
        "state": "Jarkhand",
        "country": "India",
        "limit": 5
    }
    
    try:
        result = Nearbyhospitals().crew().kickoff(inputs=inputs)
        print("Hospitals found:")
        if isinstance(result, list):
            for i, hospital in enumerate(result, start=1):
                print(f"{i}. {hospital}")
        else:
            print(result)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


if __name__ == "__main__":
    run()
