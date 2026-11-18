import pytest
from fastapi import HTTPException

import serve

PERSON = {"age": 32, "workclass": "Federal-gov", "education": "Prof-school", "education-num": 15,
          "marital-status": "Married-civ-spouse", "occupation": "Prof-specialty", "relationship": "Husband",
          "race": "White", "sex": "Male", "capital-gain": 15024, "capital-loss": 0, "hours-per-week": 50,
          "native-country": "United-States"}


def test_prediction_is_a_probability_and_a_decision():
    answer = serve.predict(PERSON)
    assert 0 <= answer["probability"] <= 1
    assert answer["above_50K"] == (answer["probability"] >= answer["threshold"])


def test_unknown_field_is_refused():
    with pytest.raises(HTTPException):
        serve.predict({"age": 30, "nonsense": 1})
