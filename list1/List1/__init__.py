from graph import *
from dijkstra_time import *
from a_star_time import *
from a_star_change import *


graph = load_graph_from_csv('connection_graph.csv')

start = time.time()

a_star_change(graph, 'KRZYKI', 'PL. GRUNWALDZKI', '22:30:00')

end = time.time()

total_time = end - start
print("\n" + str(total_time))
