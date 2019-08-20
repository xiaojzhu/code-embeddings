def bubbleSort(self, arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

def selection_sort(arr):        
    for i in range(len(arr)):
        minimum = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[minimum]:
                minimum = j
        arr[minimum], arr[i] = arr[i], arr[minimum]
        
# def selection_sort(temp):        
#     for i in range(len(temp)):
#         minimum = i
#         for j in range(i + 1, len(temp)):
#             if temp[j] < temp[minimum]:
#                 minimum = j
#         temp[minimum], temp[i] = temp[i], temp[minimum]
                
# def selection_sort(aa):        
#     for i in range(len(aa)):
#         minimum = i
#         for j in range(i + 1, len(aa)):
#             if aa[j] < aa[minimum]:
#                 minimum = j
#         aa[minimum], aa[i] = aa[i], aa[minimum]
        
# #initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
# def ss():
#     for i in range(self.k):
#         self.centroids[i] = data[i]

#     for i in range(self.max_iterations):
#         self.classes = {}
#         for i in range(self.k):
#             self.classes[i] = []

#         #find the distance between the point and cluster; choose the nearest centroid
#         for features in data:
#             distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
#             classification = distances.index(min(distances))
#             self.classes[classification].append(features)
#     previous = dict(self.centroids)
#     #average the cluster datapoints to re-calculate the centroids
#     for classification in self.classes:
#         self.centroids[classification] = np.average(self.classes[classification], axis = 0)