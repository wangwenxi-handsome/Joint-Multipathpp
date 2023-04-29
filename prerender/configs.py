from vectorizer import SegmentAndAgentSequenceRender


NCloseSegAndValidAgentRenderer = {
    "class": SegmentAndAgentSequenceRender,
    "n_closest_segments": 128,
    "max_agent_num": 32,
    "drop_segments": 6,
}


def get_vectorizer_config(vectorizer_name):
    if vectorizer_name == "NCloseSegAndValidAgentRenderer":
        return NCloseSegAndValidAgentRenderer
    raise ValueError(f"{vectorizer_name} is not supported")