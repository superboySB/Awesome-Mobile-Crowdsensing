import os

import numpy as np
import movingpandas
import pandas as pd
from ray.rllib.utils.framework import try_import_torch
from branca.element import CssLink, Figure, JavascriptLink, MacroElement
from jinja2 import Template

from warp_drive.utils.common import get_project_root

# np.seterr(invalid='ignore')
torch, nn = try_import_torch()

# from datasets.KAIST.env_config import BaseEnvConfig
# from shapely.geometry import *

# tmp_config = BaseEnvConfig()


# def inPoly(polygon, x, y):
#     pt = (x, y)
#     line = LineString(polygon)
#     point = Point(pt)
#     polygon = Polygon(line)
#     return polygon.contains(point)


# def iscrosses(line1, line2):
#     if LineString(line1).crosses(LineString(line2)):
#         return True
#     return False


# def crossPoly(square, x1, y1, x2, y2):
#     our_line = LineString([[x1, y1], [x2, y2]])
#     line1 = LineString([square[0], square[2]])
#     line2 = LineString([square[1], square[3]])
#     if our_line.crosses(line1) or our_line.crosses(line2):
#         return True
#     else:
#         return False


# def judge_collision(new_robot_px, new_robot_py, old_robot_px, old_robot_py):
#     if tmp_config.env.no_fly_zone is None:
#         return False

#     for square in tmp_config.env.no_fly_zone:
#         if inPoly(square, new_robot_px, new_robot_py):
#             return True
#         if crossPoly(square, new_robot_px, new_robot_py, old_robot_px, old_robot_py):
#             return True
#     return False
direction_map_dict = {
    0: "Stop",
    1: "E",
    2: "W",
    3: "N",
    4: "S",
    5: "NE",
    6: "SE",
    7: "NW",
    8: "SW",
}

speed_map_dict = {
    0: 18,
    1: 12,
    2: 6
}
def get_border(ur, lf):
    upper_left = [lf[0], ur[1]]
    upper_right = [ur[0], ur[1]]
    lower_right = [ur[0], lf[1]]
    lower_left = [lf[0], lf[1]]

    coordinates = [
        upper_left,
        upper_right,
        lower_right,
        lower_left,
        upper_left
    ]

    geo_json = {"type": "FeatureCollection",
                "properties": {
                    "lower_left": lower_left,
                    "upper_right": upper_right
                },
                "features": []}

    grid_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates],
        }
    }

    geo_json["features"].append(grid_feature)

    return geo_json


def traj_to_timestamped_geojson(index, trajectory: movingpandas.Trajectory, car_num, drone_num, color,
                                connect_line=False, fix_target=False, is_emergency=False, zero_shot_start=None):
    assert zero_shot_start is not None
    point_gdf = trajectory.df.copy()
    point_gdf["previous_geometry"] = point_gdf["geometry"].shift()
    point_gdf["time"] = point_gdf.index
    point_gdf["previous_time"] = point_gdf["time"].shift()
    point_gdf.loc[point_gdf["time"].iloc[0], "previous_geometry"] = point_gdf['geometry'].iloc[0]
    point_gdf.loc[point_gdf["time"].iloc[0], "previous_time"] = point_gdf['time'].iloc[0]
    # pd.Timedelta(days=0, hours=0, minutes=15))
    features = []
    is_anti_goal = False
    if isinstance(color, list):
        colors = color
    else:
        colors = None
    # for Point in GeoJSON type
    first_row = point_gdf.iloc[0]
    for i, row in enumerate(point_gdf.itertuples(index=False)):
        if fix_target:
            current_point_coordinates = [first_row.geometry.xy[0][0], first_row.geometry.xy[1][0]]
            previous_point_coordinates = [first_row.geometry.xy[0][0], first_row.geometry.xy[1][0]]
        else:
            current_point_coordinates = [row.geometry.xy[0][0], row.geometry.xy[1][0]]
            previous_point_coordinates = [row.previous_geometry.xy[0][0], row.previous_geometry.xy[1][0]]
        current_time = [row.time.isoformat()]
        previous_time = [row.previous_time.isoformat()]

        if 0 > row.id >= (-car_num):
            radius = 3  # 125(5 units)
            opacity = 0.05
            popup_html = f'<h4> (Car) Agent {car_num + drone_num - index - 1}</h4>' + \
                         f"<p style='font-size:14px;'>Location: ({int(row.x)},{int(row.y)})</p>" + \
                         f"<p style='font-size:14px;'>Timestamp: {i}</p>" + \
                         f"<p style='font-size:14px;'>Reward: {row.reward:.4f} </p>" + \
                         f"<p style='font-size:14px;'>Energy: {row.energy}J </p>"

            # f'<p>raw coord: {current_point_coordinates}</p>' + \
            # f'<p>grid coord: ({row.x},{row.y})</p>' + \
            # f'<p>dist coord: ({row.x_distance}m, {row.y_distance}m)</p>' + \
        elif -(car_num + drone_num) <= row.id < (-car_num):
            radius = 3  # 125(5 units)
            opacity = 1
            popup_html = f'<h4> (Drone) Agent {car_num + drone_num - index - 1}</h4>' + \
                         f"<p style='font-size:14px;'>Location: ({int(row.x)},{int(row.y)})</p>" + \
                         f"<p style='font-size:14px;'>Timestamp: {i}</p>" + \
                         f"<p style='font-size:14px;'>Reward: {row.reward:.4f} </p>" + \
                         f"<p style='font-size:14px;'>Energy: {row.energy}J </p>" + \
                         f"<p style='font-size:14px;'>Action: {direction_map_dict[row.direction]} </p>"
            if hasattr(row, "speed"):
                popup_html += f"<p style='font-size:14px;'>Speed: {speed_map_dict[row.speed]}m/s </p>"
        elif row.id < -(car_num + drone_num):
            radius = 4
            opacity = 1
            is_anti_goal = True
            popup_html = f'<h4> Anti-Goal {int(row.id) - drone_num - car_num}</h4>' + \
                         f"<p style='font-size:14px;'>Location: ({int(row.x)},{int(row.y)})</p>"
        else:
            if is_emergency:
                radius = 16
                if row.creation_time < i:
                    if not row.coverage:
                        opacity = (i - row.creation_time) / row.threshold
                    else:
                        opacity = 0
                else:
                    opacity = 0
                popup_html = f'<h4> Emergency {int(row.id) - zero_shot_start}</h4>' + \
                             f"<p style='font-size:14px;'>Location: ({int(row.x)},{int(row.y)})</p>" + \
                             f"<p style='font-size:14px;'>Creation Time: {row.creation_time} </p>" + \
                             f"<p style='font-size:14px;'>Delay: {int(row.aoi)} </p>" + \
                             f"<p style='font-size:14px;'>Allocation: Agent {int(row.allocation)}</p>"
            else:
                radius = 4
                opacity = 1
                popup_html = f'<h4> Surveillance {int(row.id)}</h4>' + \
                             f"<p style='font-size:14px;'>Location: ({int(row.x)},{int(row.y)})</p>" + \
                             f"<p style='font-size:14px;'>AoI: {int(row.aoi)} </p>"

        if connect_line:
            feature_dict = create_linestring_feature([previous_point_coordinates, current_point_coordinates],
                                                     [previous_time[0], current_time[0]],
                                                     color=color, caption=popup_html, opacity=opacity, radius=radius)
        else:
            if colors is not None:
                current_color = colors[i]
            else:
                current_color = color
            icon = 'circle'
            feature_dict = create_point_feature(current_color,
                                                current_point_coordinates,
                                                current_time, opacity,
                                                popup_html, radius, icon)
        features.append(feature_dict)
    return features


def create_point_feature(color, current_point_coordinates, current_time,
                         opacity, popup_html, radius, icon='circle'):
    feature_dict = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": current_point_coordinates,
        },
        "properties": {
            "times": current_time,
            'popup': popup_html,
            "icon": icon,  # point
            "iconstyle": {
                'fillColor': color,
                'fillOpacity': opacity,  # 透明度
                'stroke': 'true',
                'radius': radius,
                'weight': 1 if opacity > 0 else 0
            },

            "style": {  # line
                "color": color,
            },
            "code": 11,
        },
    }
    # if icon == 'triangle':
    #     feature_dict["properties"].update({
    #         "icon": "custom",  # point icon
    #         "iconstyle": {
    #             'iconUrl': os.path.join(get_project_root(), "envs",
    #                                     "crowd_sim", "triangle.svg"),  # URL or path to the custom icon
    #             'iconSize': [30, 30],  # Size of the icon
    #             'iconAnchor': [15, 30],  # Anchor point of the icon
    #             'popupAnchor': [0, -30],  # Anchor point for the popup
    #             'fillColor': color,
    #             'fillOpacity': opacity,  # transparency
    #             'stroke': 'true',
    #             'radius': radius,
    #             'weight': 1 if opacity > 0 else 0
    #         },
    #     })
    return feature_dict


def create_linestring_feature(coordinates, dates, color, caption=None, opacity=1, radius=5):
    feature_dict = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates,
        },
        "properties": {
            "times": dates,
            "icon": 'circle',  # point
            "iconstyle": {
                'fillColor': color,
                'fillOpacity': opacity,  # 透明度
                'stroke': 'true' if opacity > 0 else 'false',
                'radius': radius if opacity > 0 else 0,
                'weight': 1 if opacity > 0 else 0,
            },

            "style": {  # line
                "color": color,
            },
            "code": 11,
        },
    }
    if caption is not None:
        feature_dict["properties"]["popup"] = caption
    return feature_dict


def create_MultiPoint_feature(coordinates, dates, color, caption=None, opacity=1, radius=5):
    feature_dict = {
        "type": "Feature",
        "geometry": {
            "type": "MultiPoint",
            "coordinates": coordinates,
        },
        "properties": {
            "times": dates,
            "icon": 'circle',  # point
            "iconstyle": {
                'fillColor': color,
                'fillOpacity': opacity,  # 透明度
                'stroke': 'true',
                'radius': radius,
                'weight': 1 if opacity > 0 else 0,
            },

            "style": {  # line
                "color": color,
            },
            "code": 11,
        },
    }
    if caption is not None:
        feature_dict["properties"]["popup"] = caption
    return feature_dict


# if __name__ == "__main__":
#     print(judge_collision(new_robot_px=6505, new_robot_py=5130,
#                           old_robot_px=6925, old_robot_py=5130))
def map_values_to_levels(arr: np.ndarray, num_levels=8):
    # Create an array of equally spaced levels from 0 to 1
    levels = np.linspace(0, 1, num_levels + 1)

    # Map each value in the input array to the nearest level
    mapped_values = np.interp(arr, levels[:-1], levels[1:])

    return mapped_values


def generate_hotspot_circle(size, radius):
    center = (size - 1) / 2  # Center of the array

    # Create an empty array
    hotspot_array = np.zeros((size, size))

    # Calculate the Gaussian-like values based on distance from the center
    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            if distance <= radius:
                # Use a Gaussian-like function to assign values
                hotspot_array[i, j] = np.exp(-distance ** 2 / (2 * (radius / 2) ** 2))

    return hotspot_array


class JsButton(MacroElement):
    """
    Button that executes a javascript function.
    Parameters
    ----------
    title : str
         title of the button, may contain html like
    function : str
         function to execute, should have format `function(btn, map) { ... }`

    See https://github.com/prinsherbert/folium-jsbutton.
    """
    _template = Template("""
        {% macro script(this, kwargs) %}
        L.easyButton(
            '<span>{{ this.title }}</span>',
            {{ this.function }}
        ).addTo({{ this.map_name }});
        {% endmacro %}
        """)

    def __init__(self, title='', function="""
        function(btn, map){
            alert('no function defined yet.');
        }
    """):
        super(JsButton, self).__init__()
        self.title = title
        self.function = function

    def add_to(self, m):
        self.map_name = m.get_name()
        super(JsButton, self).add_to(m)

    def render(self, **kwargs):
        super(JsButton, self).render()

        figure = self.get_root()
        assert isinstance(figure, Figure), (
            'You cannot render this Element if it is not in a Figure.')

        manual_link = JavascriptLink('https://cdn.jsdelivr.net/npm/leaflet-easybutton@2/src/easy-button.js')
        manual_link.code = open(os.path.join(get_project_root(), 'envs', 'crowd_sim', 'easy-button.js')).read()
        figure.header.add_child(
            manual_link,  # noqa
            name='Control.EasyButton.js'
        )

        figure.header.add_child(
            CssLink('https://cdn.jsdelivr.net/npm/leaflet-easybutton@2/src/easy-button.css'),  # noqa
            name='Control.EasyButton.css'
        )

        figure.header.add_child(
            CssLink('https://use.fontawesome.com/releases/v5.3.1/css/all.css'),  # noqa
            name='Control.FontAwesome.css'
        )


def get_emergency_labels(matched_obs, status_dim):
    emergencies = matched_obs[..., status_dim:status_dim + 2]
    emergency_x, emergency_y = emergencies[..., 0], emergencies[..., 1]
    # digitize emergency (x,y) to 4 bins
    num_bins = 4
    bins = np.linspace(0, 1, num_bins + 1)
    emergency_x = np.digitize(emergency_x, bins) - 1
    emergency_y = np.digitize(emergency_y, bins) - 1
    labels = emergency_x * num_bins + emergency_y
    return labels


def generate_quadrant_labels(agents_pos, emergencies_pos, batch_size):
    """
    Compute the quadrant labels for each agent-emergency pair.
    The labels are as follows:
    1: (+, +)
    2: (-, +)
    3: (-, -)
    4: (+, -)

    """
    difference = emergencies_pos - agents_pos
    if isinstance(difference, np.ndarray):
        difference = torch.from_numpy(difference)
    # Compute the signs of the elements in some_array
    quadrant_x = torch.sign(difference[:, 0]).to(torch.int32)
    quadrant_y = torch.sign(difference[:, 1]).to(torch.int32)
    quadrant_labels = torch.zeros(batch_size, dtype=torch.int32)
    quadrant_labels[(quadrant_x == 1) & (quadrant_y == 1)] = 1
    quadrant_labels[(quadrant_x == -1) & (quadrant_y == 1)] = 2
    quadrant_labels[(quadrant_x == -1) & (quadrant_y == -1)] = 3
    quadrant_labels[(quadrant_x == 1) & (quadrant_y == -1)] = 4
    # special, invalid emergency label.
    quadrant_labels[(emergencies_pos[..., 0] == 0) & (emergencies_pos[..., 1] == 0)] = 0
    return quadrant_labels
