from prerender.vectorizer import SegmentAndAgentSequenceRender


NCloseSegAndValidAgentRenderer = {
    "class": SegmentAndAgentSequenceRender,
    "normalize": False,
    "segment_filtering": {
        "policy": "n_closest_segments",
        "n_closest_segments": 128,
    },
    "agent_filtering": {
        "max_agent_num": 32,
    },
    "drop_segments": 6,
}


def get_vectorizer_config(vectorizer_name):
    if vectorizer_name == "NCloseSegAndValidAgentRenderer":
        return NCloseSegAndValidAgentRenderer
    raise ValueError(f"{vectorizer_name} is not supported")