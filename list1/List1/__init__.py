from graph import *
from dijkstra_time import *
from a_star_time import *
from a_star_change import *


graph = load_graph_from_csv('connection_graph.csv')

start = time.time()

path = a_star_change(graph, 'Trzebnicka', 'Jaworowa', '12:30:00')
end = time.time()

x = graph.get_edges_from_path_time('12:30:00', path)
for edge in x:
    print("{:<10} {:<10} {:<30} {:<10} {:<20}".format(edge.line, edge.departure_time, edge.start_stop,
                                                      edge.arrival_time, edge.end_stop))

total_time = end - start
print("\n" + str(total_time))
