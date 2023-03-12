import csv
from datetime import datetime


def load_graph_from_csv(filename):
    graph: Graph = Graph()

    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            edge_id = row['id']
            company = row['company']
            line = row['line']
            departure_time = row['departure_time']
            arrival_time = row['arrival_time']
            start_stop = row['start_stop']
            end_stop = row['end_stop']
            start_lat = float(row['start_stop_lat'])
            start_lon = float(row['start_stop_lon'])
            end_lat = float(row['end_stop_lat'])
            end_lon = float(row['end_stop_lon'])

            graph.add_edge(start_stop, end_stop, edge_id, company, line, departure_time, arrival_time, start_lat,
                           start_lon, end_lat, end_lon)

    return graph


def convert_time_and_compare(source_time, target_time):
    target_time = datetime.strptime(target_time, "%H:%M:%S").time()
    source_time = datetime.strptime(source_time, "%H:%M:%S").time()
    return source_time <= target_time


class Edge:
    def __init__(self, start_stop, end_stop, edge_id, company, line, departure_time, arrival_time, start_lat, start_lon,
                 end_lat, end_lon):
        self.start_stop = start_stop
        self.end_stop = end_stop
        self.edge_id = edge_id
        self.company = company
        self.line = line
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        self.start_lat = start_lat
        self.start_lot = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon


class Node:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y


class Graph:
    edge_list = []
    node_list = set()

    def __init__(self):
        pass

    def add_edge(self, start_stop, end_stop, edge_id, company, line, departure_time, arrival_time, start_lat, start_lon,
                 end_lat, end_lon):
        self.edge_list.append(
            Edge(start_stop, end_stop, edge_id, company, line, departure_time, arrival_time, start_lat, start_lon,
                 end_lat, end_lon))
        self.node_list.add(Node(start_stop, start_lat, start_lon))
        self.node_list.add(Node(end_stop, end_lat, end_lon))

    def neighbors(self, start_stop):
        neighbors = set()
        for edge in self.edge_list:
            if edge.start_stop == start_stop:
                neighbors.add(edge.end_stop)
        return list(neighbors)

    def neighbors_lines(self, start_stop):
        neighbors = set()
        list_lines = []

        for edge in self.edge_list:
            if edge.start_stop == start_stop and edge.line not in list_lines:
                neighbors.add(edge)
                list_lines.append(edge.line)
        return list(neighbors)

    def cost_time(self, start, neighbor, current_time):
        temp_edge_list = []
        for edge in self.edge_list:
            if edge.start_stop == start \
                    and edge.end_stop == neighbor \
                    and convert_time_and_compare(current_time, edge.departure_time):
                temp_edge_list.append(edge)

        if temp_edge_list:
            return min(temp_edge_list, key=lambda edge2: edge2.arrival_time).arrival_time
        else:
            return None

    def get_edges_from_path_time(self, current_time, path):
        curr = current_time
        final_list = []
        for index, item in enumerate(path):
            if index + 1 < len(path):
                temp_edge_list = []
                for edge in self.edge_list:

                    if edge.start_stop == path[index] \
                            and edge.end_stop == path[index + 1] \
                            and convert_time_and_compare(curr, edge.departure_time):
                        temp_edge_list.append(edge)

                final_list.append(min(temp_edge_list, key=lambda edge2: edge2.arrival_time))
                curr = final_list[-1].arrival_time
                print(curr)
        return final_list

    def get_node_by_name(self, name):
        for node in self.node_list:
            if node.name == name:
                return node
        return None
