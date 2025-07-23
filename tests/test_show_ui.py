import webbrowser
import trackio


def test_init_log_show(temp_db, monkeypatch):
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: None)
    run = trackio.init(project="proj", name="run")
    trackio.log({"x": 1})
    trackio.show(project="proj")
    run.finish()
    trackio.ui.demo.close()



