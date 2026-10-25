"""Runs the dashboard headlessly with Streamlit's own test tool and checks it renders."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent / "streamlit_app.py")


def run(app=None):
    app = app or AppTest.from_file(APP, default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    return app


def test_dashboard_renders_with_the_bundled_transactions():
    app = run()
    assert app.title[0].value == "🔍 FraudLens"
    labels = [m.label for m in app.metric]
    assert "Transactions checked" in labels
    assert "Frauds caught" in labels


def test_every_detector_and_threshold_renders():
    app = run()
    for detector in app.selectbox[0].options:
        app.selectbox[0].set_value(detector)
        app.slider[0].set_value(0.3)
        run(app)


def test_filters_render():
    app = run()
    for option in app.radio[1].options:
        app.radio[1].set_value(option)
        run(app)
