import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy
import scipy.spatial.distance as ssd
# import re
from collections import Counter
from itertools import combinations
from scipy import cluster
from scipy.cluster.hierarchy import dendrogram, linkage
from unidecode import unidecode


# metoda za razdelitev besed v trojčke
def kmers(s, k=3):
    for i in range(len(s) - k + 1):
        yield s[i:i + k]


def sum_help(x, group, dist):
    s = 0
    for el in group:
        s += dist[x, el]
    return s


# metoda za izračunanje povprečne silhuide razvrstitve
def avg_silhueta(groups, distances):
    silhuete = 0
    no_member = 0
    for i, group in enumerate(groups):  # sprehodimo se po vseh skupinah
        for j, member in enumerate(group):  # sprehodimo se po članih v skupinah
            a = 0
            b = float('Inf')
            for z, g in enumerate(groups):
                if z == i:
                    if len(g) > 1:
                        a = sum_help(member, g, distances) / (len(g) - 1)  # če smo v svoji skupini
                    else:
                        a = 0
                else:
                    b_tmp = sum_help(member, g, distances) / len(g)
                    if b_tmp < b:
                        b = b_tmp
            sil = (b - a) / max(a, b)
            silhuete += sil
            no_member += 1

    return silhuete / no_member


def get_distances(data):
    d = len(data)
    distances = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            distances[i, j] = cosine_similarity(data[i], data[j])

    return distances


def k_medoids(data, noRepeat, n=100, k=5):
    # data - podatki o jezikih
    # n - maksimalno število iteracij
    # k - število skupin
    # noRepeat - število ponavitev
    d = len(data)
    distances = get_distances(data)

    res = []
    for _ in range(noRepeat):
        centroids = random.sample(range(d), k)
        for _ in range(n):
            groups = [[] for _ in range(k)]
            for x in range(d):
                tmp_min = []
                for el in centroids:
                    tmp_min.append(distances[x, el])
                inx = tmp_min.index(min(tmp_min))
                groups[inx].append(x)

            new_centroids = []
            for group in groups:
                minimum = float('Inf')
                min_el = float('inf')
                for el in group:
                    tmp_list = []
                    for i in group:
                        tmp_list.append(distances[el, i])
                    s = sum(tmp_list)
                    if s < minimum:
                        minimum = s
                        min_el = el
                new_centroids.append(min_el)

            if sorted(new_centroids) == sorted(centroids):
                break
            else:
                centroids = copy.deepcopy(new_centroids)
        silhueta = avg_silhueta(groups, distances)
        res.append(copy.deepcopy([silhueta, centroids, groups]))
    return res


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# izračun kosinusne podobnosti med dvema slovarjema trojk (uporabimo presek)
def cosine_similarity(d1, d2):
    inter_el = set(d1.keys()).intersection(set(d2.keys()))
    v1 = []
    v2 = []
    for el in inter_el:
        v1.append(d1[el])
        v2.append(d2[el])
    return round(angle(v1, v2), 5)


besedila = []
jeziki = []
trans = str.maketrans('', '', '.!?,;_-0123456789()\n\t\r')
for filename in os.listdir(os.getcwd() + "\\jeziki"):
    if filename[-4:] == ".txt":
        f = open(os.getcwd() + "\\jeziki\\" + filename, "rt", encoding="utf8").read()
        # f = re.sub(r'\b\w{1,3}\b', '', f)
        f = f.lower().translate(trans)
        f = unidecode(f)
        trojcki = list(kmers(f))
        besedila.append(Counter(trojcki))
        jeziki.append(filename[:-4])

res = k_medoids(besedila, 1000)
s_max_i = 0
s_min_i = 0
max_s = 0
min_s = 10
for i in range(len(res)):
    if res[i][0] > max_s:
        max_s = res[i][0]
        s_max_i = i
    if res[i][0] < min_s:
        min_s = res[i][0]
        s_min_i = i
print("Največja silhueta:" + str(max_s))
print("Skupine:")
for i, group in enumerate(res[s_max_i][2]):
    s = "Skupina " + str(i + 1) + ": "
    for el in group:
        s += jeziki[el] + " "
    print(s)

print()
print("Najmanjša silhueta:" + str(min_s))
print("Skupine:")
for i, group in enumerate(res[s_min_i][2]):
    s = "Skupina " + str(i + 1) + ": "
    for el in group:
        s += jeziki[el] + " "
    print(s)

# izris histograma
silhuete = []
for i in range(len(res)):
    silhuete.append(res[i][0])

plt.hist(silhuete, normed=True, bins=10)


# NAPOVEDOVANJE
def get_distances2(data, x):
    d = len(data)
    distances = np.zeros(d)

    for i in range(d):
        distances[i] = cosine_similarity(data[i], x)

    return distances


def compute_possibility(distances, jeziki):
    distances = [math.pow(x, 10) for x in distances]
    ordered = sorted(distances)
    utezeno = [1 / x for x in ordered]
    koef = 100 / sum(utezeno)

    res = []
    s = 0
    for i in range(3):
        ind = distances.index(ordered[i])
        s += utezeno[i] * koef
        res.append((jeziki[ind], utezeno[i] * koef))
        ordered[i] = float("Inf")

    return res


print("#########################")
print("NAPOVED JEZIKOV:")
for filename in os.listdir(os.getcwd() + "\\besedila"):
    if filename[-4:] == ".txt":
        f = open(os.getcwd() + "\\besedila\\" + filename, "rt", encoding="utf8").read()
        # f = re.sub(r'\b\w{1,3}\b', '', f)
        f = f.lower().translate(trans)
        f = unidecode(f)
        trojcki = list(kmers(f))
        distance = get_distances2(besedila, Counter(trojcki))
        res = compute_possibility(distance, jeziki)
        print("Besedilo: " + filename)
        print(res)

y = ssd.squareform(get_distances(besedila))

Z = scipy.cluster.hierarchy.linkage(y)
cutree = cluster.hierarchy.cut_tree(Z, n_clusters=5)
hier = [[] for x in range(5)]

for i, el in enumerate(cutree):
    hier[el[0]].append(i)

print()
print("Hierarhične klustering :")
for i, group in enumerate(hier):
    s = "Skupina " + str(i + 1) + ": "
    for language in group:
        s += jeziki[language] + " "
    print(s)

# Novičarske strani
print()
print("Testiranje na novicah:")
besedila = []
jeziki = []
trans = str.maketrans('', '', '.!?,;_-0123456789()\n\t\r')
for filename in os.listdir(os.getcwd() + "\\novice"):
    if filename[-4:] == ".txt":
        f = open(os.getcwd() + "\\novice\\" + filename, "rt", encoding="utf8").read()
        # f = re.sub(r'\b\w{1,3}\b', '', f)
        f = f.lower().translate(trans)
        f = unidecode(f)
        trojcki = list(kmers(f))
        besedila.append(Counter(trojcki))
        jeziki.append(filename[:-4])

res = k_medoids(besedila, 1000)
s_max_i = 0
s_min_i = 0
max_s = 0
min_s = 10
for i in range(len(res)):
    if res[i][0] > max_s:
        max_s = res[i][0]
        s_max_i = i
    if res[i][0] < min_s:
        min_s = res[i][0]
        s_min_i = i
print("Največja silhueta:" + str(max_s))
print("Skupine:")
for i, group in enumerate(res[s_max_i][2]):
    s = "Skupina " + str(i + 1) + ": "
    for el in group:
        s += jeziki[el] + " "
    print(s)

print()
print("Najmanjša silhueta:" + str(min_s))
print("Skupine:")
for i,group in enumerate(res[s_min_i][2]):
    s = "Skupina " + str(i + 1) + ": "
    for el in group:
        s += jeziki[el] + " "
    print(s)

plt.show()
