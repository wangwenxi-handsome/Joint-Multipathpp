import numpy as np


class Renderer:
    def render(self, data):
        pass
    
    @staticmethod
    def filter_valid(item, valid_array):
        return item[valid_array.flatten() > 0]

    @staticmethod
    def get_filter_valid_roadnetwork_keys():
        filter_valid_roadnetwork = [
            "roadgraph_samples/xyz", "roadgraph_samples/id", "roadgraph_samples/type",
            "roadgraph_samples/valid", "roadgraph_samples/dir"]
        return filter_valid_roadnetwork

    @staticmethod
    def get_filter_valid_anget_history():
        result = []
        key_with_different_timezones = ["x", "y", "speed", "bbox_yaw", "valid"]
        common_keys = [
            "state/id", "state/is_sdc", "state/type", "state/current/width", "state/current/length"]
        for key in key_with_different_timezones:
            for zone in ["past", "current", "future"]:
                result.append(f"state/{zone}/{key}")
        result.extend(common_keys)
        return result


class SegmentFilteringPolicy:
    def __init__(self, config):
        self._config = config

    def _select_n_closest_segments(self, segments, types):
        distances = np.linalg.norm(segments, axis=-1).min(axis=-1)
        n_closest_segments_ids = np.argpartition(
            distances, self._config["n_closest_segments"])[:self._config["n_closest_segments"]]
        return segments[n_closest_segments_ids], types[n_closest_segments_ids].flatten()

    def _select_segments_within_radius(self, segments, types):
        distances = np.linalg.norm(segments, axis=-1).min(axis=-1)
        closest_segments_selector = distances < self._config["segments_filtering_radius"]
        return segments[closest_segments_selector], types[closest_segments_selector].flatten()
    
    def filter(self, segments, types):
        if self._config["policy"] == "n_closest_segments":
            return self._select_n_closest_segments(segments, types)
        if self._config["policy"] == "within_radius":
            return self._select_segments_within_radius(segments, types)
        raise Exception(f"Unknown segment filtering policy {self._config['policy'] }")


class AgentFilteringPolicy:
    def __init__(self, config):
        self._config = config

    def _get_target_agents(self, data):
        return data["state/tracks_to_predict"] > 0 or data["state/is_sdc"] == 1
    
    def _get_current_available_agents(self, data):
        return np.squeeze(data["state/current/valid"]) > 0
    
    def filter(self, data):
        # Returns np.array of shape [N], which represents seleted agent.
        target_valid = self._get_target_agents(data)
        target_num = np.sum(target_valid)
        assert target_num <= self._config["max_agent_num"]
        current_available_valid = self._get_current_available_agents(data)

        # the more available history trajectory, the more available information
        sorted_agent = np.argsort(np.sum(-data["state/past/valid"], axis=-1))
        i = 0
        while(target_num < self._config["max_agent_num"]):
            if (current_available_valid[sorted_agent[i]] and ~target_valid[sorted_agent[i]]):
                i += 1
                target_valid[sorted_agent[i]] = True
        return target_valid


class SegmentAndAgentSequenceRender(Renderer):
    def __init__(self, config):
        self._config = config
        self.n_segment_types = 21
        self._segment_filter = SegmentFilteringPolicy(self._config["segment_filtering"])
        self._agent_filter = AgentFilteringPolicy(self._config["agent_filtering"])
    
    def _preprocess_data(self, data):
        # get valid roadnetwork
        valid_roadnetwork_selector = data["roadgraph_samples/valid"]
        for key in Renderer.get_filter_valid_roadnetwork_keys():
            data[key] = Renderer.filter_valid(data[key], valid_roadnetwork_selector)
        # get valid agent
        agents_with_any_validity_selector = self._agent_filter.filter(data)
        for key in Renderer.get_filter_valid_anget_history():
            data[key] = Renderer.filter_valid(data[key], agents_with_any_validity_selector)

    def _prepare_roadnetwork_info(self, data):
        # Returns np.array of shape [N, 2, 2]
        # 0 dim: N - number of segments
        # 1 dim: the start and the end of a segment
        # 2 dim: (x, y)
        # and
        # ndarray of segment types
        node_xyz = data["roadgraph_samples/xyz"][:, :2]
        node_id = data["roadgraph_samples/id"].flatten()
        node_type = data["roadgraph_samples/type"]
        result = []
        segment_types = []
        for polyline_id in np.unique(node_id):
            polyline_nodes = node_xyz[node_id == polyline_id]
            polyline_type = node_type[node_id == polyline_id][0]
            if len(polyline_nodes) == 1:
                polyline_nodes = np.array([polyline_nodes[0], polyline_nodes[0]])
            if "drop_segments" in self._config:
                selector = np.arange(len(polyline_nodes), step=self._config["drop_segments"])
                if len(polyline_nodes) <= self._config["drop_segments"]:
                    selector = np.array([0, len(polyline_nodes) - 1])
                selector[-1] = len(polyline_nodes) - 1
                polyline_nodes = polyline_nodes[selector]
            polyline_start_end = np.array(
                [polyline_nodes[:-1], polyline_nodes[1:]]).transpose(1, 0, 2)
            result.append(polyline_start_end)

            segment_types.extend([polyline_type] * len(polyline_start_end))
        result = np.concatenate(result, axis=0)
        assert len(segment_types) == len(result), \
            f"Number of segments {len(result)} doen't match the number of types {len(segment_types)}"
        return {
            "segments": result,
            "segment_types": np.array(segment_types)}
    
    def _split_past_and_future(self, data, key):
        history = np.concatenate(
            [data[f"state/past/{key}"], data[f"state/current/{key}"]], axis=1)[..., None]
        future = data[f"state/future/{key}"][..., None]
        return history, future
    
    def _prepare_agent_history(self, data):
        # (n_agents, 11, 2)
        preprocessed_data = {}
        preprocessed_data["history/xy"] = np.array([
            np.concatenate([data["state/past/x"], data["state/current/x"]], axis=1),
            np.concatenate([data["state/past/y"], data["state/current/y"]], axis=1)
        ]).transpose(1, 2, 0)
        # (n_agents, 80, 2)
        preprocessed_data["future/xy"] = np.array(
            [data["state/future/x"], data["state/future/y"]]).transpose(1, 2, 0)
        # (n_agents, 11, 1), (n_agents, 80, 1)
        for key in ["speed", "bbox_yaw", "valid"]:
            preprocessed_data[f"history/{key}"], preprocessed_data[f"future/{key}"] = \
                self._split_past_and_future(data, key)
        # (n_agents,) and (n_agents, 1)
        for key in ["state/id", "state/is_sdc", "state/type", "state/current/width",
                "state/current/length"]:
            preprocessed_data[key.split('/')[-1]] = data[key]
        # string
        preprocessed_data["scenario_id"] = data["scenario/id"]
        return preprocessed_data
    
    def _transfrom_to_agent_coordinate_system(self, coordinates, shift, yaw):
        # coordinates
        # dim 0: number of agents / number of segments for road network
        # dim 1: number of history points / (start_point, end_point) for segments
        # dim 2: x, y
        yaw = -yaw
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array(((c, -s), (s, c))).reshape(2, 2)
        transformed = np.matmul((coordinates - shift), R.T)
        return transformed

    def _filter_closest_segments(self, position, segments, types):
        # This method works only with road segments in agent-related coordinate system
        assert position.shape == (2,)
        assert len(segments.shape) == 3
        assert segments.shape[1] == segments.shape[2] == 2
        assert len(segments) == len(types), \
            f"n_segments={len(segments)} must match len_types={len(types)}"
        return self._segment_filter.filter(position, segments, types)
    
    def _compute_closest_point_of_segment(self, position, segments):
        assert position.shape == (2,)
        assert len(segments.shape) == 3
        assert segments.shape[1] == segments.shape[2] == 2
        A, B = segments[:, 0, :], segments[:, 1, :]
        M = B - A
        t = ((position[None, :] - A) * M).sum(axis=-1) / ((M * M).sum(axis=-1) + 1e-6)
        clipped_t = np.clip(t, 0, 1)[:, None]
        closest_points = A + clipped_t * M
        return closest_points
    
    def _generate_segment_embeddings(self, position, yaw, segments, types):
        # closest point vector(length and heading)
        closest_points = self._compute_closest_point_of_segment(position, segments)
        r_norm = np.linalg.norm(closest_points - position[None, :], axis=-1, keepdims=True)
        r_unit_vector = (closest_points - position[None, :]) / (r_norm + 1e-6)

        # segment vector(length and heading)
        segment_end_minus_start = segments[:, 1, :] - segments[:, 0, :]
        segment_end_minus_start_norm = np.linalg.norm(
            segment_end_minus_start, axis=-1, keepdims=True)
        segment_unit_vector = segment_end_minus_start / (segment_end_minus_start_norm + 1e-6)
        segment_end_minus_r_norm = np.linalg.norm(
            segments[:, 1, :] - closest_points, axis=-1, keepdims=True)

        segment_type_ohe = np.eye(self.n_segment_types)[types]
        resulting_embeddings = np.concatenate([
            np.tile(position, (segments.shape[0], 1)), np.tile(yaw, (segments.shape[0], 1)), 
            r_norm, r_unit_vector, segment_unit_vector, segment_end_minus_start_norm, 
            segment_end_minus_r_norm, segment_type_ohe], axis=-1)
        return resulting_embeddings[:, None, :]

    def render(self, data):
        self._preprocess_data(data)
        road_network_info = self._prepare_roadnetwork_info(data)
        agent_history_info = self._prepare_agent_history(data)

        # get ego coordinate
        ego_id = np.where(data["state/is_sdc"] == 1)[0][0]
        target_id = list(np.where(data["state/tracks_to_predict"] == 1)[0])
        # check agent order is [target_agent, ego, other_agent]
        assert target_id == list(range(max(target_id) + 1))
        assert max(target_id) == ego_id or max(target_id) == ego_id - 1
        current_agent_scene_shift = agent_history_info["history/xy"][ego_id][-1]
        current_agent_scene_yaw = agent_history_info["history/bbox_yaw"][ego_id][-1]

        # convert pos and yaw to the ego coordinate
        current_scene_agents_coordinates_history = self._transfrom_to_agent_coordinate_system(
            agent_history_info["history/xy"], current_agent_scene_shift, current_agent_scene_yaw)
        current_scene_agents_coordinates_future = self._transfrom_to_agent_coordinate_system(
            agent_history_info["future/xy"], current_agent_scene_shift, current_agent_scene_yaw)
        current_scene_agents_yaws_history = \
            agent_history_info["history/bbox_yaw"] - current_agent_scene_yaw
        current_scene_agents_yaws_future = \
            agent_history_info["future/bbox_yaw"] - current_agent_scene_yaw

        # compute related polylines of target agents
        current_scene_road_network_coordinates = self._transfrom_to_agent_coordinate_system(
            road_network_info["segments"], current_agent_scene_shift, current_agent_scene_yaw)

        road_segments_embeddings = []
        for i in range(len(agent_history_info["history/xy"])):
            target_road_network_coordinates, target_road_network_types = \
                self._filter_closest_segments(
                    current_scene_agents_coordinates_history[i][-1], 
                    current_scene_road_network_coordinates, 
                    road_network_info["segment_types"]
                )
            target_road_segments_embeddings = self._generate_segment_embeddings(
                current_scene_agents_coordinates_history[i][-1], 
                current_scene_agents_yaws_history[i][-1],
                target_road_network_coordinates, 
                target_road_network_types
            )
            road_segments_embeddings.append(target_road_segments_embeddings)
        road_segments_embeddings = np.stack(road_segments_embeddings, axis = 0)

        # return scene_data
        scene_data = {
            # int and string
            "ego_id": ego_id,
            "target_id": max(target_id),
            "scenario_id": agent_history_info["scenario_id"].item().decode("utf-8"),
            # (1, )
            "yaw": current_agent_scene_yaw,
            # (1, 2)
            "shift": current_agent_scene_shift[None, ],
            # (n, )
            "agent_type": agent_history_info["type"].astype(int),
            "agent_id": agent_history_info["id"],
            # (n, 1)
            "width": agent_history_info["width"],
            "length": agent_history_info["length"],
            # (n, 11, -1)
            "history/xy": current_scene_agents_coordinates_history,
            "history/yaw": current_scene_agents_yaws_history,
            "history/speed": agent_history_info["history/speed"],
            "history/valid": agent_history_info["history/valid"],
            # (n, 80, -1)
            "future/xy": current_scene_agents_coordinates_future,
            "future/yaw": current_scene_agents_yaws_future,
            "future/speed": agent_history_info["future/speed"],
            "future/valid": agent_history_info["future/valid"],
            # (max_agent_num, 128, 1, -1)
            "road_network_embeddings": road_segments_embeddings,
        }
        return [scene_data]
