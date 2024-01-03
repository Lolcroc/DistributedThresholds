# Python
from __future__ import annotations
from collections import defaultdict, deque

edge_int = int
vertex_int = int

def xor_from_set(s, element):
    if element in s:
        s.remove(element)
        return True
    else:
        s.add(element)
        return False

# Same as set.discard but returns True if actually discarded
def discard_from_set(s, element):
    if element in s:
        s.remove(element)
        return True
    else:
        return False

# Will infinitely recur if graph is not a tree
# TODO remove recursion due to recursion limit
def traverse(graph, current, previous=None):
    neighbours = graph[current]

    for neighbour, leaf in neighbours.items():
        if neighbour != previous:
            yield from traverse(graph, neighbour, current)
            yield current, neighbour, leaf

def find(i, parents):
    if parents[i] != i:
        parents[i] = find(parents[i], parents)
    return parents[i]

def union(cluster, unions, parents, clusters, boundary=True):
    for c in unions:
        parents[c.root] = cluster.root
        cluster.parity ^= c.parity
        if boundary:
            cluster.boundary.update(c.boundary)
        cluster.collisions += c.collisions

        del clusters[c.root]  # Cluster c is now safe to delete

class BucketQ:
    """A queue consisting of integer slots, with each slot a a stack of items.
    
    This implementations uses dict[int, None] types instead of a set[int] for the buckets, because the insertion order
    is respected for dicts and not for sets. This is important if one adds a cluster with a smaller growing weight,
    because it will be added to the current higher bucket value and should thus be grown first.
    """
    def __init__(self):
        self._max = 0
        self._cur = 0
        self._data = defaultdict(dict)
        self._poplast = False  # If each bucket is not popped first, the error rates are influenced heavily (badly)

    def __iter__(self):
        start = 0
        while start < self._max:
            for weight in range(start, self._max + 1):
                self._cur = weight
                data = self._data[weight]
                while data:
                    if self._poplast:
                        cluster, _ = data.popitem()  # This will almost never happen
                        self._poplast = False
                    else:
                        cluster = next(iter(data))
                        del data[cluster]
                    yield cluster

            start = weight
        self._cur = 0

    def add(self, cluster):
        if cluster.parity:
            self._poplast = cluster.grow_weight < self._cur  # This almost never happens
            weight = max(self._cur, cluster.grow_weight)  # @ Niels, grow_weight is just some integer
            self._max = max(self._max, weight)
        else:
            weight = -1
        self._data[weight][cluster] = None

    def remove(self, cluster):
        if cluster.parity:
            weight = cluster.grow_weight
        else:
            weight = -1
        del self._data[weight][cluster]

    @property
    def even_clusters(self):
        return self._data[-1]

class Leaf:
    __slots__ = ("tip", "vein", "base", "root", "reverse")

    def __init__(self, tip: vertex_int, vein: edge_int):
        # Immutable, initialized before decoding
        self.tip = tip  # Outer vertex
        self.vein = vein  # Edge connecting tip and base
        self.reverse: Leaf = None  # Cyclic reference that has to be set up later

        # Mutable, used during decoding
        self.base: Leaf = None
        self.root: vertex_int = -1

    def root_path(self):
        leaf = self
        while leaf != None:
            yield leaf.vein
            leaf = leaf.base

        leaf = self.reverse.base
        while leaf != None:
            yield leaf.vein
            leaf = leaf.base
    
    def __repr__(self):
        return f"{self.__class__.__name__}(tip={self.tip}, vein={self.vein}, root={self.root})"

    def __hash__(self):
        return hash((self.tip, self.vein))

class Cluster:
    __slots__ = ("root", "parity", "boundary", "collisions")

    def __init__(self, root: vertex_int, parity: int, boundary: set[Leaf]):
        self.root = root
        self.parity = parity
        self.boundary = boundary
        self.collisions: deque[Leaf] = deque()

    def __repr__(self):
        return f"{self.__class__.__name__}(root={self.root}, parity={self.parity})"
    
    def __hash__(self):
        return self.root

    @property
    def grow_weight(self) -> int:
        return len(self.boundary)
    
    def union_weight(self) -> int:
        return len(self.boundary)
    
    def peeling_tree(self):
        tree: dict[vertex_int, dict[vertex_int, Leaf]] = defaultdict(dict)

        for leaf in self.collisions:
            u, v = leaf.root, leaf.reverse.root
            tree[u][v] = leaf
            tree[v][u] = leaf.reverse

        yield from traverse(tree, self.root)

class UFDecoder:
    def __init__(self, crystal):
        # The primal syndrome graph derives from the dual crystal
        # This is a matter of convention
        syndrome_graph = crystal.dual

        self.seedlings: dict[vertex_int, list[Leaf]] = defaultdict(list)
        self.buds: dict[Leaf, list[Leaf]] = {}

        self.a_neighbour: dict[edge_int, vertex_int] = {} # Used for input erasures

        for edge in syndrome_graph.filter_nodes(dim=1):
            v1, v2 = syndrome_graph.successors(edge)
            l1, l2 = Leaf(tip=v1, vein=edge), Leaf(tip=v2, vein=edge)
            l1.reverse, l2.reverse = l2, l1
            self.seedlings[v1].append(l2)
            self.seedlings[v2].append(l1)
            self.a_neighbour[edge] = v1

        for leaves in self.seedlings.values():
            for leaf in leaves:
                # Note to self; this used to check if the next_leaf source != leaf source
                # However this strategy is bogged if the syndrome graph has multi-edges.
                # We filter based on the leaf vein to ensure proper boundary growth on
                # syndromes graphs with multi-edges.
                self.buds[leaf] = list(next_leaf for next_leaf in self.seedlings[leaf.tip] if next_leaf.vein != leaf.vein)

        self.seedlings = dict(self.seedlings)
    
    def leaves_from_root(self, source: vertex_int):
        for l in self.seedlings[source]:
            l.base = None
            l.root = source
            yield l

    def leaves_from_leaf(self, source: Leaf):
        for l in self.buds[source]:
            l.base = source
            l.root = source.root
            yield l

    def decode(self, syndromes, erasures: set = None):
        # Key = root
        clusters = dict()
        parents = dict()

        # Key = bucket
        sorted_clusters = BucketQ()

        # Support
        support = set()

        # Union-Find
        for s in syndromes:
            cluster = Cluster(s, 1, set(self.leaves_from_root(s)))
            sorted_clusters.add(cluster)
            clusters[s] = cluster
            parents[s] = cluster.root

        if erasures:
            clusters_grown = set()

            # First grow existing clusters along the sublattice defined by erasures
            # This step also removes those edges from the erasures
            # We are careful not to grow a cluster twice whenever this process
            # creates a union using the clusters_grown set
            to_grow = list(clusters.values())
            for cluster in to_grow:
                if cluster not in clusters_grown:
                    self.grow_erasures(cluster, clusters, parents, erasures, sorted_clusters, clusters_grown)

            # There are still erasures left without an attached cluster
            # Create 0-parity clusters for these and grow them like before
            while erasures:
                edge = erasures.pop()
                root = self.a_neighbour[edge]

                cluster = Cluster(root, 0, set(self.leaves_from_root(root)))
                sorted_clusters.add(cluster)
                clusters[root] = cluster
                parents[root] = cluster.root

                self.grow_erasures(cluster, clusters, parents, erasures, sorted_clusters, clusters_grown)

        for cluster in sorted_clusters:
            self.grow(cluster, clusters, parents, support, sorted_clusters)

        # Peeling
        correction = set()
        syn = syndromes.copy()

        # print(f"Pre-peeling clusters: {clusters}")
        # print(f"Pre-peeling even clusters: {sorted_clusters.even_clusters}")
        for cluster in sorted_clusters.even_clusters:  # TODO loop over clusters
            for u, v, leaf in cluster.peeling_tree():
                if v in syn:
                    correction ^= set(leaf.root_path())
                    syn.remove(v)

                    if u in syn:
                        syn.remove(u)
                    else:
                        syn.add(u)

            del clusters[cluster.root]

        return correction

    def grow(self, cluster, clusters, parents, support, sorted_clusters):
        bound = cluster.boundary
        unions = [cluster]   # TODO index by root
        collisions = deque()  # Linked list

        for leaf in bound.copy():
            vein = leaf.vein
            tip = leaf.tip

            if xor_from_set(support, vein):  # Union
                bound.remove(leaf)
                if tip not in parents:  # New cluster. Do a cheap union
                    # Add leaves going from this leaf
                    bound.update(self.leaves_from_leaf(leaf))

                    # Set Union-Find parent to this root
                    parents[tip] = cluster.root
                    # find(cluster.root, parents)  # Dummy call for profiling
                else:  # Existing cluster. Expensive union
                    other_root = find(parents[tip], parents)
                    other = clusters[other_root]

                    if other not in unions:
                        unions.append(other)
                        collisions.append(leaf)

                        sorted_clusters.remove(other)  # TODO index by root

                    other.boundary.remove(leaf.reverse)
        
        if len(unions) > 1:
            cluster = max(unions, key=Cluster.union_weight)
            unions.remove(cluster)

            union(cluster, unions, parents, clusters)
            cluster.collisions += collisions
        
        sorted_clusters.add(cluster)

    def grow_erasures(self, cluster, clusters, parents, erasures, sorted_clusters, clusters_grown):
        sorted_clusters.remove(cluster)
        unions = [cluster]   # TODO index by root
        collisions = deque()  # Linked list

        bound = cluster.boundary
        new_bound = set()

        while bound:
            leaf = bound.pop()
            vein = leaf.vein
            tip = leaf.tip

            if discard_from_set(erasures, vein):  # Union
                if tip not in parents:  # New cluster. Do a cheap union
                    # Add leaves going from this leaf
                    bound.update(self.leaves_from_leaf(leaf))

                    # Set Union-Find parent to this root
                    parents[tip] = cluster.root
                    # find(cluster.root, parents)  # Dummy call for profiling
                else:  # Existing cluster. Expensive union
                    other_root = find(parents[tip], parents)
                    other = clusters[other_root]

                    if other not in unions:
                        unions.append(other)
                        collisions.append(leaf)

                        sorted_clusters.remove(other)  # TODO index by root
                        bound.update(other.boundary)

                    bound.remove(leaf.reverse)
            else:
                new_bound.add(leaf)

        clusters_grown.update(unions) # Unioned cluters shouldn't grow again
        
        if len(unions) > 1:
            cluster = max(unions, key=Cluster.union_weight)
            unions.remove(cluster)

            union(cluster, unions, parents, clusters, False)
            
            cluster.collisions += collisions

        cluster.boundary = new_bound
        sorted_clusters.add(cluster)

# 3x3x3 cubic (on dual crystal)
# erasures = {160, 193, 100, 104, 86, 157, 95}
# syndromes = {32, 50, 52, 31}
