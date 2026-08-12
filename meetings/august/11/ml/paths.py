import os

HERE = os.path.dirname(os.path.abspath(__file__))


def fixture_dir():
    return os.path.join(HERE, "data_runs", "fixture")


def repo_meeting_dir(month, day):
    # ml -> 11 -> august -> meetings, so three levels up, not two
    meetings = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
    return os.path.join(meetings, month, day)
