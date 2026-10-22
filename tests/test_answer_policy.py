import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generation.answer_generator import (
    requires_live_environment_data,
)


def test_current_kali_ip_requires_live_data():
    question = "What is the current IP address assigned to my Kali machine?"

    assert requires_live_environment_data(question)


def test_current_user_ip_requires_live_data():
    question = "What is my current IP address?"

    assert requires_live_environment_data(question)


def test_static_document_question_does_not_trigger_guard():
    question = "How can I display an IP address using Linux?"

    assert not requires_live_environment_data(question)


def test_historical_document_question_does_not_trigger_guard():
    question = "What IP address appears in the book's example?"

    assert not requires_live_environment_data(question)