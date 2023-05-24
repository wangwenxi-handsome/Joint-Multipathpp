from vectorizer import SegmentAndAgentSequenceRender


N16CloseSegAndValidAgentRenderer = {
    "class": SegmentAndAgentSequenceRender,
    "n_closest_segments": 128,
    "select_all_agent": False,
    "max_agent_num": 16,
    "drop_segments": 6,
}


N32CloseSegAndValidAgentRenderer = {
    "class": SegmentAndAgentSequenceRender,
    "n_closest_segments": 128,
    "select_all_agent": False,
    "max_agent_num": 32,
    "drop_segments": 6,
}

NCloseSegAndValidAgentRenderer = {
    "class": SegmentAndAgentSequenceRender,
    "select_all_agent": True,
    "n_closest_segments": 128,
    "drop_segments": 6,
}


def get_vectorizer_config(vectorizer_name):
    if vectorizer_name == "N16CloseSegAndValidAgentRenderer":
        return N16CloseSegAndValidAgentRenderer
    if vectorizer_name == "N32CloseSegAndValidAgentRenderer":
        return N32CloseSegAndValidAgentRenderer
    if vectorizer_name == "NCloseSegAndValidAgentRenderer":
        return NCloseSegAndValidAgentRenderer
    raise ValueError(f"{vectorizer_name} is not supported")