import json
import os
import pandas as pd


def find_components(graph):
    visited = set()
    components = []

    def dfs(node, comp):
        visited.add(node)
        comp.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, comp)

    for node in graph:
        if node not in visited:
            comp = []
            dfs(node, comp)
            components.append(comp)

    return components


if __name__ == "__main__":
    UNDIRECTED_GRAPH = "undirected_graph.json"
    LOSTBENCH_PATH = "../LoStBench/datalake/"
    TESTCASES_PATH = "test_case.json"

    with open(UNDIRECTED_GRAPH, "r", encoding = "utf-8") as f:
        test_graph = json.load(f)
    
    with open(TESTCASES_PATH, "r", encoding = "utf-8") as f:
        test_cases = json.load(f)
    
    count = 0
    # This triple level nested loop proves that there is no one to many case coming from connected components generation
    for file, adj_list in test_graph.items():
        columns = pd.read_csv(os.path.join(LOSTBENCH_PATH, file), nrows = 0).columns.to_list()
        columns = set(columns)

        components = find_components(test_graph[file])
        
        for lst in components:
            if len(columns.intersection(set(lst))) > 1:
                print(set(lst))
            
            for item in test_cases[file]:
                cols2 = set(pd.read_csv(os.path.join(LOSTBENCH_PATH, item), nrows = 0).columns.to_list())
                if len(cols2.intersection(set(lst))) > 1:
                    print(set(lst), file, item)

        count += 1
        print(f"Finished for {count} cases out of {len(test_graph)}")
        # break


