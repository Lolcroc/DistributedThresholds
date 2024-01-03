from collections import defaultdict

# class NoClusterObjBucketQ:
#     def __init__(self):
#         self._max = 0
#         self._cur = 0
#         self._data = defaultdict(OrderedSet)

#     def __iter__(self):
#         prevmax = 0
#         while prevmax < self._max:
#             for key in range(prevmax, self._max + 1):
#                 self._cur = key
#                 data = self._data[key]
#                 while data:
#                     yield data.pop()

#             prevmax = key

#     def add(self, key, value):
#         key = max(self._cur, key)
#         self._max = max(self._max, key)
#         self._data[key].add(value)

#     def remove(self, key, value):
#         self._data[key].remove(value)

## OLD CODE
#
# def gen():
#     i = 0
#     while i < 1000:
#         n = (yield i)
#         if n is not None:
#             i += n
#         i += 1

# g = gen()

# test = None
# while True:
#     test = g.send(test)
#     print(test)
