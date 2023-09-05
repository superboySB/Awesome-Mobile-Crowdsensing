import networkx as nx

class Roadmap():

    def __init__(self, dataset_str):
        # 注意x和y是pygame坐标系，原点在左上角，竖直方向为纬度和x，水平方向为经度和y。这与去年暑假不同

        self.dataset_str = dataset_str
        self.map_props = get_map_props()

        self.lower_left = get_map_props()[dataset_str]['lower_left']
        self.upper_right = get_map_props()[dataset_str]['upper_right']

        if dataset_str == 'NCSU':
            self.max_dis_x = 3255.4913305859623
            self.max_dis_y = 2718.3945272795013
        elif dataset_str == 'purdue':
            self.max_dis_x = 1671.8995666382975
            self.max_dis_y = 1221.4710883988212
        elif dataset_str == 'KAIST':
            self.max_dis_x = 2100.207579392558
            self.max_dis_y = 2174.930950809533
        elif dataset_str =='Rome':
            self.max_dis_x = 5997.844092533406
            self.max_dis_y = 6152.200368709555

    def lonlat2pygamexy(self, lon, lat):
        '''
        将某个点的经纬度转换为pygame坐标轴下的x、y坐标.
        经度为y，纬度为x，原点在左上角
        参数可以是标量也可以是np.array
        '''    
        x = self.max_dis_x * (lon - self.lower_left[0]) / (self.upper_right[0] - self.lower_left[0])
        y = self.max_dis_y * (self.upper_right[1]- lat) / (self.upper_right[1] - self.lower_left[1])
        return x, y

    def pygamexy2lonlat(self, x, y):
        lon = x/self.max_dis_x  * (self.upper_right[0] - self.lower_left[0]) + self.lower_left[0]
        lat =  -y/self.max_dis_y * (self.upper_right[1] - self.lower_left[1])  + self.upper_right[1]
        return lon, lat

def get_map_props():
    map_props = {
        'NCSU':
            {
                'lower_left': [-78.6988, 35.7651],  # lon, lat
                'upper_right': [-78.6628, 35.7896]
            },
        'purdue':
            {
                'lower_left': [-86.93, 40.4203],
                'upper_right': [-86.9103, 40.4313]
            },
        'KAIST':
            {
                'lower_left': [127.3475, 36.3597],
                'upper_right': [127.3709, 36.3793]
            },
        'Rome':
            {
                 'lower_left': [12.4523, 41.865],
                'upper_right': [12.5264, 41.919]
            }
    }
    return map_props


def get_sub_graph(graph: nx.Graph, sub_g: list):
        if len(sub_g) == 0:
            return graph
        remove_list = []
        for u, v, data in graph.edges(keys=False, data=True):
            if v in sub_g or u in sub_g:
                remove_list.append([u, v])
        graph.remove_edges_from(remove_list)
        return graph

