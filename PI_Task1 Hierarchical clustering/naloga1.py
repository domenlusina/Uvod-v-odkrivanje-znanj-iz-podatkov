import copy
import csv
import itertools
import math
import numpy as np
from tkinter import *

np.set_printoptions(threshold=np.nan)
np.seterr(divide='ignore', invalid='ignore')

noCountries = 47


# funkcija za pretvorbo števil iz strinov v int
def string_to_int(i):
    return int(i) if len(i) > 0 else None


# funkcija za deljenje - če je prisotna vrednost None vrne None
def divide(a, b):
    if a is None or b is None:
        return None
    else:
        return a / b


def n_max(arr, n):
    k = sum(np.isnan(arr))
    indices = arr.ravel().argsort()[-(n + k):]
    indices = (np.unravel_index(i, arr.shape) for i in indices)
    res = []
    for i in indices:
        if not math.isnan(arr[i]):
            res.append([arr[i], i])
    return res


def n_min(arr, n):
    indices = arr.ravel().argsort()[:n]
    indices = (np.unravel_index(i, arr.shape) for i in indices)
    return [[arr[i], i] for i in indices]


# funkcija za dodelitev novih ocen vektorju ocen neke države (countriesVotes),
# z vektorjem števila glasov votesCounter
# in vektorjem novih ocen newGrades
def add_votes(countriesGrades, votesCounter, newGrades):
    newVotesCount = [0 if isUnknown(x) else 1 for x in newGrades]
    countriesCounter = [a + b for a, b in zip(votesCounter, newVotesCount)]

    newGrades = [0 if isUnknown(el) else el for el in newGrades]
    countriesGrades = [a + b for a, b in zip(countriesGrades, newGrades)]

    return countriesGrades, countriesCounter


def isUnknown(x):
    if x is None or math.isnan(x):
        return True
    else:
        return False


def row_distance(r1, r2):
    r = [i for i in zip(r1, r2) if i[0] is not None and i[1] is not None]
    if len(r) < 1:
        return float('Inf')
    else:
        return math.sqrt(sum((a - b) ** 2 for a, b in r) / len(r))


# Cluster je struktura, ki predstavlja posamezno skupino
# ima kazalca na levega in desnega sina in hrani razdaljo združenih sinov in vse države označene z številkami
class Cluster:
    def __init__(self, distance, countries):
        self.left = None
        self.right = None
        self.countries = countries
        self.distance = distance
        self.y = 0

    def set_left(self, left):
        self.left = left
        self.countries.extend(left.countries)

    def set_right(self, right):
        self.right = right
        self.countries.extend(right.countries)


class HierarchicalClustering:
    def __init__(self, distances, countries, dist_type="max"):
        self.distances = distances
        self.countries = countries
        self.dist_type = dist_type
        self.clusters = [Cluster(0, [y]) for y in range(len(countries))]

    def cluster_distance(self, c1, c2):
        # izračunamo razdaljo med dvema skupinama (minimalno, maksimalno ali povprečno)
        # distance - matrika razdalj med državami
        # c1, c2 - seznam držav predstavljene s števili

        # zapišemo vse možne kombinacij držav iz c1 z državam iz c2
        comb = list(itertools.product(c1, c2))
        dists = [self.distances[el] for el in comb]

        # odstranimo tiste, ki imajo vrednost Inf
        dists = list(filter((float('Inf')).__ne__, dists))

        # če ne moremo primerjati dveh vektorjov vrnemo neskončno
        if len(dists) == 0:
            return float('Inf')
        if self.dist_type == "avg":
            return sum(dists) / (len(dists))
        elif self.dist_type == "min":
            return min(dists)
        elif self.dist_type == "max":
            return max(dists)
        else:
            print("Unknown linkage type")
            return 0

    def join_closest_clusters(self):
        # funkcija join_closest cluster združi, dve najbližji skupini na podlagi Evklidske razdalje
        dol = len(self.clusters) - 1
        min_i1 = -1
        min_i2 = -1
        dist = float('Inf')
        # pridobimo razdaljo med najbližjima skupinama in indeksa le teh
        for i in range(dol):
            for j in range(dol - i):
                d = self.cluster_distance(self.clusters[i].countries, self.clusters[i + j + 1].countries);
                if d < dist:
                    min_i1 = i
                    min_i2 = i + j + 1
                    dist = d

        if dol > 0:
            # združimo najbližji skupini v novo skupino in odstranimo stari dve, ter novo dodamo v seznam
            joined_cluster = Cluster(dist, [])
            joined_cluster.set_left(self.clusters[min_i1])
            joined_cluster.set_right(self.clusters[min_i2])
            self.clusters.pop(min_i2)
            self.clusters.pop(min_i1)
            self.clusters.append(joined_cluster)
        else:
            print("Premalo skupin !!")

    def run(self):
        if len(self.clusters) >= 1:
            for r in range(len(self.clusters) - 1):
                self.join_closest_clusters()
        else:
            print("There are no clusters!")


data = []
f = open("eurovision-final.csv", "rt", encoding="latin1")

# preberemo csv datoteko
for line in csv.reader(f):
    lineArray = [''.join(x.strip().split(', ')) for x in line]
    data.append(lineArray)

# odstranimo prazne stolpce
data = np.array(data)
for i in reversed(range(len(data[0]))):
    if np.all(data[:, i] == ''):
        data = np.delete(data, np.s_[i], axis=1)

# pridobimo seznam vseh drzav
countries = data[0][-noCountries:]

# Serbia & Montenegro je včasih zapisan kot Serbia and Montenegro
countries = [w.replace('&', 'and') for w in countries]
data = data[1:]
songs = list(zip(*data))[5]

# slovar, ki vrne indeks drzave oz pesmi
indexCountry = dict((el, i) for i, el in enumerate(countries))
indexSongs = dict((el, i) for i, el in enumerate(songs))

votesSum = np.zeros((noCountries, noCountries))
votesCounter = np.zeros((noCountries, noCountries))
votesAvg = np.zeros((noCountries, noCountries))

# ustvarimo dve matrika- ena predstavlja seštevek vseh ocen skozi leta, druga število glasovanj
for line in data:
    d = line[-noCountries:]
    key = indexCountry[line[1]]
    d = list(map(string_to_int, d))
    votesSum[key], votesCounter[key] = add_votes(votesSum[key], votesCounter[key], d)

# izračunamo povprečje
votesAvg = np.divide(votesSum, votesCounter)

votesAvg = votesAvg.transpose()

# izluščimo samo podatke o ocenah
data = list(zip(*data))[-noCountries:]
data = [list(map(string_to_int, x)) for x in data]

# ustvarimo matriko razdalj za vse države (dimenzije 47*47)
distance_matrix = np.zeros((noCountries, noCountries))
for i in range(noCountries):
    for j in range(noCountries):
        distance_matrix[i, j] = row_distance(data[i], data[j])

hc = HierarchicalClustering(distance_matrix, countries)
# poženemo algoritem za clustering
hc.run()
# dobimo drevo, ki opisuje hierarhijo
tree = hc.clusters


# funkcija, ki pridobi seznam globin vozlišč oz listov (1.,3.,5.... element so listi, 2.,4.,6. elementi so vozlišča)
def izpis_seznam(tree, deepth):
    x = []
    if tree.left is not None and len(tree.left.countries) > 1:
        a = izpis_seznam(tree.left, deepth + 1)
        x.extend(a)
    else:
        x.append(deepth + 1)
    x.extend([deepth])
    if tree.right is not None and len(tree.right.countries) > 1:
        b = izpis_seznam(tree.right, deepth + 1)
        x.extend(b)
    else:
        x.append(deepth + 1)
    return x


# izpis tekstovnega dendograma
te = izpis_seznam(tree[0], 0)
for i, x in enumerate(te):
    str_te = x * "    "
    if i % 2 == 0:
        str_te = str_te + "----" + countries[tree[0].countries[int(i / 2)]]
    else:
        str_te = str_te + "----|"
    print(str_te)


# GRAFIČNI IZRIS

# dodelimo vrstico vsakemu vozlišču oziroma listu
# lista vsebujejo vrednosti npr. 0 in 1, njun starš se potem nahaja med njima 0.5* višina vrstice
def set_y(tree, mx):
    if tree.left is None and tree.right is None:
        tree.y = mx
        mx = mx + 1
    else:
        mx = set_y(tree.left, mx)
        y1 = tree.left.y
        mx = set_y(tree.right, mx)
        y2 = tree.right.y
        tree.y = (y1 + y2) / 2

    return mx


mx = set_y(tree[0], 0)


# vrne seznam kjer vsak element predstavlja
# [razdaljo, (y1 in y2 koordinato vodoravne črte), (dolžino zgornje oz spodnje vodoravne črte)]
# vsakega vozlišča v drevesu
def izpis_razdalj(tree):
    seznam = []
    d1, d2, y1, y2 = (0,) * 4
    if tree.left is not None:
        d1 = tree.left.distance
        y1 = tree.left.y
    if tree.right is not None:
        d2 = tree.right.distance
        y2 = tree.right.y

    if tree.left is not None and len(tree.left.countries) > 1:
        a = izpis_razdalj(tree.left)
        seznam.extend(a)

    seznam.append([tree.distance, (y1, y2), (tree.distance - d1, tree.distance - d2)])
    if tree.right is not None and len(tree.right.countries) > 1:
        b = izpis_razdalj(tree.right)
        seznam.extend(b)
    return seznam


grafi = izpis_razdalj(tree[0])
ranges = [x[0] for x in grafi]
max_grafi = max(ranges)
min_grafi = min(ranges)

# določimo črto na dendogramu, ki določi število skupin
ranges_sorted = ranges
ranges_sorted.sort(reverse=True)
max_dist = 0
spodnja_meja = 0
zgornja_meja = 0
for i in range(len(ranges_sorted) - 1):
    tmp_dist = ranges_sorted[i] - ranges_sorted[i + 1]
    if tmp_dist > max_dist and ranges_sorted[i] < max_grafi and min_grafi < ranges_sorted[i + 1]:
        max_dist = tmp_dist
        zgornja_meja = ranges_sorted[i]
        spodnja_meja = ranges_sorted[i + 1]


def doloci_skupine(tree, spodnja):
    x = []
    if tree is not None and tree.distance <= spodnja:
        return [tree]
    if tree.left is not None:
        a = doloci_skupine(tree.left, spodnja)
        x.extend(a)
    if tree.right is not None:
        b = doloci_skupine(tree.right, spodnja)
        x.extend(b)
    return x


skupine = doloci_skupine(tree[0], spodnja_meja)
skupineGlasovi = np.zeros([len(skupine), noCountries])
stevecGlasov = np.zeros([len(skupine), noCountries])
skupineAvg = np.zeros([len(skupine), noCountries])

for i, skupina in enumerate(skupine):
    for sk in skupina.countries:
        skupineGlasovi[i], stevecGlasov[i] = add_votes(skupineGlasovi[i], stevecGlasov[i], votesAvg[sk])

skupineAvg = np.divide(skupineGlasovi, stevecGlasov)

for i in range(len(skupineAvg)):
    print("#" * 30)
    print("Skupina " + str(i + 1) + ":")
    str0 = "Države: "
    str1 = "Največ glasov:"
    str2 = "Najmanj glasov:"
    max_arr = n_max(skupineAvg[i], 5)
    for el in skupine[i].countries:
        str0 = str0 + " " + countries[el] + "|"
    for el in max_arr:
        str1 = str1 + " " + countries[el[1][0]] + " " + str(el[0]) + "|"
    min_arr = n_min(skupineAvg[i], 5)
    for el in min_arr:
        str2 = str2 + " " + countries[el[1][0]] + " " + str(el[0]) + "|"
    print(str0)
    print(str1)
    print(str2)

h_line = 20  # višina "vrstice" na canvasu
width = 800
height = 960

root = Tk()
frame = Frame(root, width=width, height=height)
frame.pack(side=LEFT, expand=True, fill=BOTH)
canvas = Canvas(frame, bg='#FFFFFF', width=width, height=height, scrollregion=(0, 0, width, height))

hbar = Scrollbar(frame, orient=HORIZONTAL)
hbar.pack(side=BOTTOM, fill=X)
hbar.config(command=canvas.xview)
vbar = Scrollbar(frame, orient=VERTICAL)
vbar.pack(side=RIGHT, fill=Y)
vbar.config(command=canvas.yview)

canvas.config(width=width, height=height)
canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

# izpis teksta držav
for i, country in enumerate(countries):
    canvas.create_text(610, (i + 1) * h_line, text=countries[tree[0].countries[i]], anchor="w")

# izris ločitvene črte
x = 600 * (1 - (zgornja_meja + spodnja_meja) / (2 * max_grafi)) + 10
canvas.create_line(x, 0, x, height, width=3, fill='red')

# izris črt
for i in range(len(grafi)):
    x = 600 * (1 - grafi[i][0] / max_grafi) + 10
    y1 = grafi[i][1][1] * h_line + h_line
    y2 = grafi[i][1][0] * h_line + h_line

    # navpična črta
    canvas.create_line(x, y1, x, y2)
    # prva vodoravna črta
    x1 = 600 * ((grafi[i][2][1]) / max_grafi)
    canvas.create_line(x, y1, x + x1, y1)
    # druga vodoravna črta
    x2 = 600 * ((grafi[i][2][0]) / max_grafi)
    canvas.create_line(x, y2, x + x2, y2)
canvas.pack()
root.mainloop()
